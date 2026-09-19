from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from permission.permission import PermissionManager  # noqa: E402
from tools import build_bwrap_command, is_sandbox_permission_error, run_bash  # noqa: E402


class AutoModePermissionTests(unittest.TestCase):
    def test_auto_mode_approves_write_and_bash_tools(self) -> None:
        manager = PermissionManager(mode="auto")

        self.assertEqual("allow", manager.check("write_file", {"path": "a.txt"}).behavior)
        self.assertEqual("allow", manager.check("bash", {"command": "printf ok"}).behavior)
        self.assertEqual("allow", manager.check("bash", {"command": "printf ok | wc -c"}).behavior)

    def test_auto_mode_still_denies_high_risk_bash(self) -> None:
        manager = PermissionManager(mode="auto")

        self.assertEqual("deny", manager.check("bash", {"command": "sudo true"}).behavior)


class BubblewrapCommandTests(unittest.TestCase):
    def test_workspace_setting_binds_only_workspace_writable(self) -> None:
        with (
            patch("tools.shutil.which", return_value="/usr/bin/bwrap"),
            patch.object(config, "SANDBOX_SETTING", "workspace"),
            patch.object(config, "SANDBOX_NETWORK_ACCESS", False),
        ):
            args = build_bwrap_command("touch result.txt")

        workspace = str(Path.cwd().resolve())
        self.assertEqual(["--ro-bind", "/", "/"], args[3:6])
        self.assertIn("--unshare-net", args)
        self.assertIn(["--bind", workspace, workspace], [args[i:i + 3] for i in range(len(args) - 2)])
        self.assertEqual(["--chdir", workspace, "--", "/bin/bash", "-lc", "touch result.txt"], args[-6:])

    def test_missing_bwrap_fails_closed(self) -> None:
        with patch("tools.shutil.which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "refusing to run"):
                build_bwrap_command("true")

    def test_read_setting_does_not_add_writable_root_bind(self) -> None:
        with (
            patch("tools.shutil.which", return_value="/usr/bin/bwrap"),
            patch.object(config, "SANDBOX_SETTING", "read"),
            patch.object(config, "SANDBOX_NETWORK_ACCESS", True),
        ):
            args = build_bwrap_command("true")

        self.assertNotIn("--bind", args)
        self.assertNotIn("--unshare-net", args)

    def test_all_setting_makes_root_writable(self) -> None:
        with (
            patch("tools.shutil.which", return_value="/usr/bin/bwrap"),
            patch.object(config, "SANDBOX_SETTING", "all"),
            patch.object(config, "SANDBOX_NETWORK_ACCESS", True),
        ):
            args = build_bwrap_command("true")

        self.assertEqual(["--bind", "/", "/"], args[3:6])


class RunBashTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_mode_uses_exec_with_bwrap(self) -> None:
        process = AsyncMock()
        process.communicate.return_value = (b"ok\n", b"")
        process.returncode = 0

        with (
            patch.object(config, "PERMISSION_MODE", "auto"),
            patch("tools.build_bwrap_command", return_value=["bwrap", "--", "bash"]),
            patch("tools.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as create_exec,
            patch("tools.asyncio.create_subprocess_shell", AsyncMock()) as create_shell,
        ):
            output = await run_bash("printf ok")

        self.assertEqual("ok", output)
        create_exec.assert_awaited_once()
        create_shell.assert_not_awaited()

    async def test_permission_failure_can_be_retried_without_sandbox(self) -> None:
        sandbox_process = AsyncMock()
        sandbox_process.communicate.return_value = (b"", b"touch: Read-only file system\n")
        sandbox_process.returncode = 1
        host_process = AsyncMock()
        host_process.communicate.return_value = (b"host retry succeeded\n", b"")
        host_process.returncode = 0

        with (
            patch.object(config, "PERMISSION_MODE", "auto"),
            patch("tools.build_bwrap_command", return_value=["bwrap", "--", "bash"]),
            patch("tools.asyncio.create_subprocess_exec", AsyncMock(return_value=sandbox_process)),
            patch("tools.asyncio.create_subprocess_shell", AsyncMock(return_value=host_process)) as create_shell,
            patch("tools.asyncio.to_thread", AsyncMock(return_value=True)) as ask_user,
        ):
            output = await run_bash("touch /outside")

        self.assertEqual("host retry succeeded", output)
        ask_user.assert_awaited_once()
        create_shell.assert_awaited_once()

    async def test_declined_unsandboxed_retry_preserves_sandbox_error(self) -> None:
        sandbox_process = AsyncMock()
        sandbox_process.communicate.return_value = (b"", b"touch: Permission denied\n")
        sandbox_process.returncode = 1

        with (
            patch.object(config, "PERMISSION_MODE", "auto"),
            patch("tools.build_bwrap_command", return_value=["bwrap", "--", "bash"]),
            patch("tools.asyncio.create_subprocess_exec", AsyncMock(return_value=sandbox_process)),
            patch("tools.asyncio.create_subprocess_shell", AsyncMock()) as create_shell,
            patch("tools.asyncio.to_thread", AsyncMock(return_value=False)) as ask_user,
        ):
            output = await run_bash("touch /outside")

        self.assertEqual("touch: Permission denied", output)
        ask_user.assert_awaited_once()
        create_shell.assert_not_awaited()


class SandboxPermissionErrorTests(unittest.TestCase):
    def test_requires_failure_and_known_permission_marker(self) -> None:
        self.assertTrue(is_sandbox_permission_error(1, b"Read-only file system"))
        self.assertTrue(is_sandbox_permission_error(126, b"Operation not permitted"))
        self.assertFalse(is_sandbox_permission_error(0, b"Permission denied"))
        self.assertFalse(is_sandbox_permission_error(1, b"command not found"))


if __name__ == "__main__":
    unittest.main()
