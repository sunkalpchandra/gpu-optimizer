import React from "react";
import ReactDOM from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import RunView from "./pages/RunView";
import SearchTree from "./pages/SearchTree";
import KernelViewer from "./pages/KernelViewer";
import GpuMetrics from "./pages/GpuMetrics";
import Reports from "./pages/Reports";

const router = createHashRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Overview /> },
      { path: "runs/:runId", element: <RunView /> },
      { path: "runs/:runId/tree", element: <SearchTree /> },
      { path: "kernels/:candidateId", element: <KernelViewer /> },
      { path: "gpu", element: <GpuMetrics /> },
      { path: "reports", element: <Reports /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
