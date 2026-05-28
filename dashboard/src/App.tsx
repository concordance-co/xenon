import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "@/components/Shell";
import { RunsIndex } from "@/routes/RunsIndex";
import { RunDetail } from "@/routes/RunDetail";
import { ProjectReportGallery, ReportGallery } from "@/routes/ReportGallery";
import { PhaseDetailPage, ProjectDetailPage, ProjectsIndex } from "@/routes/ProjectsIndex";

export function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <Shell>
            <Navigate to="/projects" replace />
          </Shell>
        }
      />
      <Route
        path="/projects"
        element={
          <Shell>
            <ProjectsIndex />
          </Shell>
        }
      />
      <Route
        path="/projects/reports/:reportKey"
        element={
          <Shell>
            <ProjectReportGallery />
          </Shell>
        }
      />
      <Route
        path="/projects/:projectId"
        element={
          <Shell>
            <ProjectDetailPage />
          </Shell>
        }
      />
      <Route
        path="/projects/:projectId/phases/*"
        element={
          <Shell>
            <PhaseDetailPage />
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
