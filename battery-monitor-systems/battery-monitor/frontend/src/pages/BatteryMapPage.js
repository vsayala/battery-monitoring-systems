import React, { useEffect, useState } from "react";
import MapClusterHeat from "../components/MapClusterHeat";

export default function BatteryMapPage() {
  const [batteries, setBatteries] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    fetch(`${process.env.REACT_APP_API_URL}/batteries`)
      .then(res => res.json())
      .then(setBatteries);
  }, []);

  return (
    <div>
      <h2>Battery Map</h2>
      <label>
        <input type="checkbox" checked={showHeatmap} onChange={e => setShowHeatmap(e.target.checked)} />
        Show Heatmap
      </label>
      <MapClusterHeat batteries={batteries} showHeatmap={showHeatmap} />
    </div>
  );
}
