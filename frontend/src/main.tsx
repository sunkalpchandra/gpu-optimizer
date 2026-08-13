import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import { Loading } from "./components/ui";

// Route-level code splitting keeps the recharts-heavy pages out of the
// initial bundle.
const Overview = lazy(() => import("./pages/Overview"));
const RunView = lazy(() => import("./pages/RunView"));
const SearchTree = lazy(() => import("./pages/SearchTree"));
const KernelViewer = lazy(() => import("./pages/KernelViewer"));
const GpuMetrics = lazy(() => import("./pages/GpuMetrics"));
const Reports = lazy(() => import("./pages/Reports"));

const wrap = (el: React.ReactNode) => <Suspense fallback={<Loading />}>{el}</Suspense>;

const router = createHashRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: wrap(<Overview />) },
      { path: "runs/:runId", element: wrap(<RunView />) },
      { path: "runs/:runId/tree", element: wrap(<SearchTree />) },
      { path: "kernels/:candidateId", element: wrap(<KernelViewer />) },
      { path: "gpu", element: wrap(<GpuMetrics />) },
      { path: "reports", element: wrap(<Reports />) },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
