import json
from pathlib import Path
from datetime import datetime
from threading import Lock


class JsonlTracer:
    """本地 JSONL Trace 记录器"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self._lock = Lock()

    def _get_file(self, run_id: str) -> Path:
        """一个 run_id 对应一个日志文件"""

        date_dir = (
            self.log_dir
            / datetime.now().strftime("%Y-%m-%d")
        )

        date_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return date_dir / f"{run_id}.jsonl"

    @staticmethod
    def _json_default(obj):
        """处理 Pydantic、日期等不能直接 JSON 序列化的对象"""

        if hasattr(obj, "model_dump"):
            return obj.model_dump()

        return str(obj)

    def write(
        self,
        run_id: str,
        node: str,
        event: str,
        data=None
    ):
        record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "node": node,
            "event": event,
            "data": data
        }

        line = json.dumps(
            record,
            ensure_ascii=False,
            default=self._json_default
        )

        path = self._get_file(run_id)

        # 防止并发写文件互相干扰
        with self._lock:
            with path.open(
                "a",
                encoding="utf-8"
            ) as f:
                f.write(line + "\n")