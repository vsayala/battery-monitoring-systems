import React, { useEffect, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.markercluster";
import "leaflet.heat";

function MarkersCluster({ batteries }) {
  const map = useMap();
  const markerClusterGroup = useRef();
  useEffect(() => {
    if (!map) return;
    if (markerClusterGroup.current) {
      markerClusterGroup.current.clearLayers();
      map.removeLayer(markerClusterGroup.current);
    }
    const clusterGroup = L.markerClusterGroup();
    batteries.forEach(b => {
      if (b.location && b.location.lat && b.location.lng) {
        const marker = L.marker([b.location.lat, b.location.lng]);
        marker.bindPopup(
          `<b>${b.name}</b><br>Voltage: ${b.voltage}<br>Temp: ${b.temperature}<br>Charge: ${b.charge}`
        );
        clusterGroup.addLayer(marker);
      }
    });
    map.addLayer(clusterGroup);
    markerClusterGroup.current = clusterGroup;
    return () => {
      if (markerClusterGroup.current) {
        markerClusterGroup.current.clearLayers();
        map.removeLayer(markerClusterGroup.current);
      }
    };
  }, [batteries, map]);
  return null;
}

function HeatLayer({ batteries }) {
  const map = useMap();
  useEffect(() => {
    const heatData = batteries
      .filter(b => b.location && b.location.lat && b.location.lng)
      .map(b => [b.location.lat, b.location.lng, Math.max(0.1, b.charge / 100)]);
    const heat = L.heatLayer(heatData, { radius: 25, blur: 15 });
    map.addLayer(heat);
    return () => map.removeLayer(heat);
  }, [batteries, map]);
  return null;
}

export default function MapClusterHeat({ batteries, showHeatmap }) {
  const validLocs = batteries.filter(b => b.location && b.location.lat && b.location.lng);
  const center = validLocs.length
    ? [
        validLocs.reduce((sum, b) => sum + b.location.lat, 0) / validLocs.length,
        validLocs.reduce((sum, b) => sum + b.location.lng, 0) / validLocs.length
      ]
    : [0, 0];

  return (
    <MapContainer center={center} zoom={3} style={{ height: 500, width: "100%" }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {showHeatmap ? <HeatLayer batteries={batteries} /> : <MarkersCluster batteries={batteries} />}
    </MapContainer>
  );
}
