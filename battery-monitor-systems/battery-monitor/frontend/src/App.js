import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import DashboardPage from "./pages/DashboardPage";
import AlertManagementPage from "./pages/AlertManagementPage";
import AdminDeviceKeysPage from "./pages/AdminDeviceKeysPage";
import AuditLogPage from "./pages/AuditLogPage";
import BatteryMapPage from "./pages/BatteryMapPage";
import AdvancedTelemetryDashboard from "./pages/AdvancedTelemetryDashboard";
// ...other imports

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/alerts" element={<AlertManagementPage />} />
        <Route path="/admin/device-keys" element={<AdminDeviceKeysPage />} />
        <Route path="/admin/audit-logs" element={<AuditLogPage />} />
        <Route path="/batterymap" element={<BatteryMapPage />} />
        <Route path="/advanced-telemetry" element={<AdvancedTelemetryDashboard />} />
        {/* ...other routes per your previous features */}
      </Routes>
    </BrowserRouter>
  );
}
export default App;
