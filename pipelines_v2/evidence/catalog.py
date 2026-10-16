"""Read existing local mirrors without creating directories or another index."""
from pathlib import Path
import json


def local_catalog(root=None):
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.core.paths import pipelines_v2_catalog_root

    class ReadOnlyFileCatalog(FileCatalog):
        def list_workflow_runs(self, *, workflow_name=None, workflow_hash=None, status=None, limit=None):
            from pipelines_v2.workflow.records import WorkflowRunRecord
            # This is a bounded discovery window, not a new persisted index. Count
            # attempted files before filtering; a filter cannot force a full scan.
            cap = min(limit if limit is not None else 10001, 10001)
            object.__setattr__(self, 'scan_truncated', False)
            object.__setattr__(self, 'scan_files', 0)
            object.__setattr__(self, 'scan_bytes', 0)
            records = []
            for path in self._workflow_runs_root().glob('*.json'):
                if self.scan_files >= cap:
                    object.__setattr__(self, 'scan_truncated', True)
                    break
                object.__setattr__(self, 'scan_files', self.scan_files + 1)
                remaining = min(1024**2, 16 * 1024**2 - self.scan_bytes)
                with path.open('rb') as stream:
                    data = stream.read(remaining + 1)
                object.__setattr__(self, 'scan_bytes', self.scan_bytes + len(data))
                if len(data) > remaining:
                    object.__setattr__(self, 'scan_truncated', True)
                    break
                record = WorkflowRunRecord.from_dict(json.loads(data))
                if workflow_name is not None and record.workflow_name != workflow_name:
                    continue
                if workflow_hash is not None and record.workflow_hash != workflow_hash:
                    continue
                if status is not None and record.status != status:
                    continue
                records.append(record)
            records.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
            return records

        def _artifacts_root(self):
            return Path(self.root)

        def _workflow_runs_root(self):
            return Path(self.root) / 'workflow_runs'

        def _workflow_steps_root(self, run_id):
            return Path(self.root) / 'workflow_steps' / run_id

    return ReadOnlyFileCatalog(Path(root).expanduser() if root else pipelines_v2_catalog_root())
