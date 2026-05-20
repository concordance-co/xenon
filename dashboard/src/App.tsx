import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "@/components/Shell";
import { RunsIndex } from "@/routes/RunsIndex";
import { RunDetail } from "@/routes/RunDetail";
import { ReportGallery } from "@/routes/ReportGallery";

export function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <Shell>
            <Navigate to="/runs" replace />
          </Shell>
        }
      />
      <Route
        path="/runs"
        element={
          <Shell>
            <RunsIndex />
          </Shell>
        }
      />
      <Route
        path="/runs/:runId"
        element={
          <Shell>
            <RunDetail />
          </Shell>
        }
      />
      <Route
        path="/runs/:runId/reports/:artifactId"
        element={
          <Shell>
            <ReportGallery />
          </Shell>
        }
      />
      <Route
        path="*"
        element={
          <Shell>
            <div className="p-6 text-xs font-mono text-ink-400">Not found.</div>
          </Shell>
        }
      />
    </Routes>
  );
}
