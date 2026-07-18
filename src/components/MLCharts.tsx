/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend
} from 'recharts';

// Core metric interface for comparison charting
interface ComparatorMetric {
  name: string;
  value: number;
}

// 1. Histogram / Continuous Value Binning
export function HistogramChart({
  data,
  columnName,
  binsCount = 10
}: {
  data: Record<string, any>[];
  columnName: string;
  binsCount?: number;
}) {
  const chartData = useMemo(() => {
    const values = data.map(r => Number(r[columnName])).filter(v => !isNaN(v));
    if (values.length === 0) return [];
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const binWidth = range / binsCount || 1;
    
    const bins = Array.from({ length: binsCount }, (_, i) => {
      const lower = min + i * binWidth;
      const upper = lower + binWidth;
      return {
        label: `${lower.toFixed(1)} - ${upper.toFixed(1)}`,
        minVal: lower,
        maxVal: upper,
        count: 0
      };
    });

    values.forEach(v => {
      let bIdx = Math.floor((v - min) / binWidth);
      if (bIdx >= binsCount) bIdx = binsCount - 1;
      if (bIdx < 0) bIdx = 0;
      if (bins[bIdx]) bins[bIdx].count++;
    });

    return bins;
  }, [data, columnName, binsCount]);

  if (chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-400 font-mono text-sm">
        No numeric records available
      </div>
    );
  }

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
          <XAxis 
            dataKey="label" 
            angle={-15} 
            textAnchor="end" 
            tick={{ fill: '#4B5563', fontSize: 10 }}
          />
          <YAxis tick={{ fill: '#4B5563', fontSize: 11 }} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', borderRadius: '8px', border: 'none' }}
            labelStyle={{ color: '#F3F4F6', fontFamily: 'monospace', fontSize: '12px' }}
            itemStyle={{ color: '#60A5FA', fontFamily: 'monospace', fontSize: '13px' }}
          />
          <Bar dataKey="count" fill="url(#blueGrad)" radius={[4, 4, 0, 0]} name="Record Count">
            <defs>
              <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.85}/>
                <stop offset="95%" stopColor="#1D4ED8" stopOpacity={0.4}/>
              </linearGradient>
            </defs>
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 2. Interactive Scatter Plot
export function ScatterPlotChart({
  data,
  xCol,
  yCol,
  colorCol
}: {
  data: Record<string, any>[];
  xCol: string;
  yCol: string;
  colorCol?: string;
}) {
  const chartData = useMemo(() => {
    return data
      .map(row => ({
        x: Number(row[xCol]),
        y: Number(row[yCol]),
        label: colorCol ? String(row[colorCol]) : 'Record'
      }))
      .filter(item => !isNaN(item.x) && !isNaN(item.y));
  }, [data, xCol, yCol, colorCol]);

  // Group by class if colorCol is specified
  const series = useMemo(() => {
    if (!colorCol) {
      return [{ name: 'Data Point', data: chartData, color: '#3B82F6' }];
    }
    const groups: Record<string, typeof chartData> = {};
    chartData.forEach(item => {
      if (!groups[item.label]) groups[item.label] = [];
      groups[item.label].push(item);
    });

    const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
    return Object.entries(groups).map(([name, points], idx) => ({
      name,
      data: points,
      color: colors[idx % colors.length]
    }));
  }, [chartData, colorCol]);

  if (chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-400 font-mono text-sm">
        No numeric dimensions specified
      </div>
    );
  }

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name={xCol} 
            unit="" 
            tick={{ fill: '#4B5563', fontSize: 11 }}
            label={{ value: xCol, position: 'bottom', offset: 0, fill: '#6B7280', fontSize: 12 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name={yCol} 
            tick={{ fill: '#4B5563', fontSize: 11 }}
            label={{ value: yCol, angle: -90, position: 'insideLeft', offset: 5, fill: '#6B7280', fontSize: 12 }}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ backgroundColor: '#1F2937', borderRadius: '8px', border: 'none' }}
            itemStyle={{ color: '#F3F4F6', fontFamily: 'monospace', fontSize: '12px' }}
          />
          {series.map(s => (
            <Scatter 
              key={s.name} 
              name={s.name} 
              data={s.data} 
              fill={s.color} 
              shape="circle"
              line={false}
            />
          ))}
          {colorCol && <Legend verticalAlign="top" height={36}/>}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// 3. Correlation Heatmap Grid (Pearson-based relative density)
export function CorrelationHeatmap({
  columns,
  data
}: {
  columns: string[];
  data: Record<string, any>[];
}) {
  const correlationMatrix = useMemo(() => {
    const numericCols = columns.filter(col => {
      return data.every(row => {
        const v = row[col];
        return v === undefined || v === null || v === '' || !isNaN(Number(v));
      });
    }).slice(0, 10); // Limit to top 10 for grid layout aesthetics

    if (numericCols.length === 0 || data.length === 0) return { labels: [], grid: [] };

    const grid: number[][] = [];
    
    for (let i = 0; i < numericCols.length; i++) {
      grid[i] = [];
      const colX = numericCols[i];
      const valsX = data.map(r => Number(r[colX])).filter(v => !isNaN(v));
      const meanX = valsX.reduce((a, b) => a + b, 0) / valsX.length;

      for (let j = 0; j < numericCols.length; j++) {
        const colY = numericCols[j];
        const valsY = data.map(r => Number(r[colY])).filter(v => !isNaN(v));
        const meanY = valsY.reduce((a, b) => a + b, 0) / valsY.length;

        let num = 0;
        let denX = 0;
        let denY = 0;

        for (let rIdx = 0; rIdx < data.length; rIdx++) {
          const valX = Number(data[rIdx][colX]);
          const valY = Number(data[rIdx][colY]);
          if (!isNaN(valX) && !isNaN(valY)) {
            const dx = valX - meanX;
            const dy = valY - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
          }
        }

        const denominator = Math.sqrt(denX * denY);
        const corr = denominator > 0 ? num / denominator : 1;
        grid[i][j] = Math.max(-1, Math.min(1, corr));
      }
    }

    return {
      labels: numericCols,
      grid
    };
  }, [columns, data]);

  if (correlationMatrix.labels.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-400 font-mono text-sm">
        No numeric features present for correlation analysis
      </div>
    );
  }

  const getColorClass = (val: number) => {
    // Return custom color styling based on correlation values
    if (val >= 0.7) return 'bg-emerald-500 text-white';
    if (val >= 0.4) return 'bg-emerald-200 text-emerald-900';
    if (val >= 0.1) return 'bg-emerald-50 text-emerald-800';
    if (val <= -0.7) return 'bg-rose-500 text-white';
    if (val <= -0.4) return 'bg-rose-200 text-rose-900';
    if (val <= -0.1) return 'bg-rose-50 text-rose-800';
    return 'bg-gray-100 text-gray-700';
  };

  return (
    <div className="overflow-x-auto py-2">
      <div className="inline-block min-w-full align-middle">
        <table className="min-w-full border-collapse border border-gray-200 font-mono text-xs">
          <thead>
            <tr>
              <th className="p-2 border border-gray-200 bg-gray-50 text-left font-medium text-gray-600">Variables</th>
              {correlationMatrix.labels.map(lbl => (
                <th key={lbl} className="p-2 border border-gray-200 bg-gray-50 text-center font-medium text-gray-600 writing-vertical max-w-[80px] overflow-hidden text-ellipsis whitespace-nowrap">
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {correlationMatrix.labels.map((rowLbl, rowIdx) => (
              <tr key={rowLbl}>
                <td className="p-2 border border-gray-200 bg-gray-50 font-medium text-gray-700 max-w-[120px] overflow-hidden text-ellipsis whitespace-nowrap">
                  {rowLbl}
                </td>
                {correlationMatrix.labels.map((colLbl, colIdx) => {
                  const val = correlationMatrix.grid[rowIdx]?.[colIdx] ?? 0;
                  return (
                    <td 
                      key={colLbl} 
                      className={`p-3 border border-gray-200 text-center font-bold font-mono transition-all duration-150 cursor-help hover:scale-[1.05] ${getColorClass(val)}`}
                      title={`Correlation between ${rowLbl} & ${colLbl}: ${val.toFixed(3)}`}
                    >
                      {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 4. Feature Imp Importance Weights
export function FeatureImportanceBarChart({
  items
}: {
  items: Array<{ feature: string; weight: number }>;
}) {
  const chartData = useMemo(() => {
    return [...items]
      .sort((a, b) => b.weight - a.weight)
      .map(item => ({
        feature: item.feature,
        importance: Math.max(0.001, Math.round(item.weight * 1000) / 10)
      }));
  }, [items]);

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 15, right: 20, left: 10, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
          <XAxis type="number" unit="%" tick={{ fill: '#4B5563', fontSize: 11 }} />
          <YAxis 
            type="category" 
            dataKey="feature" 
            width={110} 
            tick={{ fill: '#2563EB', fontSize: 10, fontWeight: 500 }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', borderRadius: '8px', border: 'none' }}
            itemStyle={{ color: '#34D399', fontFamily: 'monospace', fontSize: '13px' }}
          />
          <Bar dataKey="importance" fill="url(#violetGrad)" radius={[0, 4, 4, 0]} name="Gini Weight (%)">
            <defs>
              <linearGradient id="violetGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.9}/>
                <stop offset="95%" stopColor="#EC4899" stopOpacity={0.7}/>
              </linearGradient>
            </defs>
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 5. Model Metrics Multi-Comparator
export function MetricsBarChart({
  modelsMetrics,
  metricKey,
  metricLabel
}: {
  modelsMetrics: Array<{ model: string; value: number }>;
  metricKey: string;
  metricLabel: string;
}) {
  const formattedData = useMemo(() => {
    return modelsMetrics.map(item => ({
      model: item.model,
      percentage: Math.round(item.value * 1000) / 10
    }));
  }, [modelsMetrics]);

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={formattedData} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F3F4F6" />
          <XAxis dataKey="model" tick={{ fill: '#4B5563', fontSize: 11 }} />
          <YAxis tick={{ fill: '#4B5563', fontSize: 11 }} unit="%" />
          <Tooltip 
            contentStyle={{ backgroundColor: '#111827', borderRadius: '8px', border: 'none' }}
            itemStyle={{ color: '#68D391', fontFamily: 'monospace' }}
          />
          <Bar dataKey="percentage" fill="url(#tealGrad)" name={`${metricLabel} (%)`} radius={[6, 6, 0, 0]}>
            <defs>
              <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#14B8A6" stopOpacity={0.85}/>
                <stop offset="95%" stopColor="#0F766E" stopOpacity={0.4}/>
              </linearGradient>
            </defs>
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
