from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict
import asyncio, logging, json, aiofiles, random

from directory import DURABLE_PATH

CRON_TRIGGER_INTERVAL = 1  # 每1秒检查一次cron任务

logger = logging.getLogger(__name__)

cron_lock = asyncio.Lock()
scheduled_jobs: Dict[str, "CronJob"] = {}
cron_queue: List["CronJob"] = []
_last_fired: Dict[str, str] = {}  # 任务上次触发的时间，job_id -> last fired minute marker (YYYY-MM-DD HH:MM)

@dataclass
class CronJob:
    id: str
    cron: str        # "0 9 * * *" (五段式 cron 表达式)
    prompt: str      # 触发时注入给 Agent 的消息
    recurring: bool  # True=周期性，False=一次性
    durable: bool    # True=写磁盘，跨会话保留

async def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """Register a new cron job. Returns CronJob or error string."""
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    async with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        await save_durable_jobs()
    logger.info(f"[cron schedule] {job.id} → {job.prompt[:40]}")
    return job

def validate_cron(cron_expr: str) -> str | None:
    """
    检查cron表达式是否合法。返回错误信息或None。

    Validate a cron expression. Returns error message or None.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None

def _validate_cron_field(field: str, low: int, high: int) -> str | None:
    """Validate a single cron field value is within [low, high]."""
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), low, high)
            if err: return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < low or a > high or b < low or b > high:
            return f"Range {field} out of bounds [{low}-{high}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < low or val > high:
        return f"Value {val} out of bounds [{low}-{high}]"
    return None

async def cancel_job(job_id: str) -> str:
    """Cancel a cron job."""
    async with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        await save_durable_jobs()
    logger.info(f"[cron cancel] {job_id}")
    return f"Cancelled {job_id}"

async def list_jobs() -> str:
    """List all scheduled cron jobs as text ."""
    jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"
        dur = "durable" if j.durable else "session"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)




def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """检查datetime是不是符合cron表达式的时间点"""
    fields: List[str] = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val: int = (dt.weekday() + 1) % 7  # Python Monday=0 → cron Sunday=0

    m: bool = _cron_field_matches(minute, dt.minute)
    h: bool = _cron_field_matches(hour, dt.hour)
    dom_ok: bool = _cron_field_matches(dom, dt.day)
    month_ok: bool = _cron_field_matches(month, dt.month)
    dow_ok: bool = _cron_field_matches(dow, dow_val)

    if not (m and h and month_ok):
        return False
    # DOM and DOW: both constrained → either matching is enough (OR)
    dom_unconstrained: bool = dom == "*"
    dow_unconstrained: bool = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok

def _cron_field_matches(field: str, value: int) -> bool:
    """Match a single cron field against a value."""
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)

async def cron_schedule_loop():
    """定时检查任务，任务触发时投递到 cron_queue"""
    while True:
        await asyncio.sleep(CRON_TRIGGER_INTERVAL)
        now: datetime = datetime.now()
        minute_marker: str = now.strftime("%Y-%m-%d %H:%M")
        async with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            logger.info(f"[cron fire] {job.id} → {job.prompt[:40]}")
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                await save_durable_jobs()
                except Exception as e:
                    logger.error(f"[cron error] {job.id}: {e}")

async def save_durable_jobs():
    """Persist durable jobs to .scheduled_tasks.json."""
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    async with aiofiles.open(DURABLE_PATH, "w") as f:
        await f.write(json.dumps(durable, indent=4))

