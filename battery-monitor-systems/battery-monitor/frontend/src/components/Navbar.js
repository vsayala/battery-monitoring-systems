import React from "react";
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav style={{ padding: 8, background: "#eee", marginBottom: 16 }}>
      <Link to="/">Dashboard</Link> |{" "}
      <Link to="/alerts">Alerts</Link> |{" "}
      <Link to="/admin/device-keys">Device Keys</Link> |{" "}
      <Link to="/admin/audit-logs">Audit Logs</Link> |{" "}
      <Link to="/batterymap">Map</Link> |{" "}
      <Link to="/advanced-telemetry">Advanced Telemetry</Link>
    </nav>
  );
}
