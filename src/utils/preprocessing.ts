/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dataset, PreprocessedDataset, PreprocessingReport, ColumnInfo } from '../types';

export interface PreprocessingOptions {
  missingStrategy: 'mean' | 'median' | 'mode' | 'drop';
  removeDuplicates: boolean;
  scalingMethod: 'none' | 'standard' | 'minmax';
  handleOutliers: 'none' | 'clip' | 'remove';
  featureSelectionCount: number; // 0 for all
}

export function preprocessDataset(
  dataset: Dataset,
  options: PreprocessingOptions,
  targetColumn?: string
): PreprocessedDataset {
  let data = JSON.parse(JSON.stringify(dataset.rawData)) as Record<string, any>[];
  const originalCount = data.length;
  
  const report: PreprocessingReport = {
    missingHandled: {},
    duplicatesRemoved: 0,
    encodedColumns: [],
    scaledColumns: [],
    outliersDetected: 0,
    selectedFeatures: []
  };

  // 1. Duplicate removal
  if (options.removeDuplicates) {
    const seen = new Set<string>();
    const uniqueData: Record<string, any>[] = [];
    for (const row of data) {
      const str = JSON.stringify(row);
      if (!seen.has(str)) {
        seen.add(str);
        uniqueData.push(row);
      }
    }
    report.duplicatesRemoved = originalCount - uniqueData.length;
    data = uniqueData;
  }

  // Calculate stats of current data (some values might be null)
  const columns = dataset.columns;
  const labelEncodings: Record<string, Record<string | number, number>> = {};
  const scalingParams: Record<string, { mean: number; std: number; min: number; max: number }> = {};

  // 2. Missing Value Handling
  for (const col of columns) {
    const colName = col.name;
    const nullCount = data.filter(r => r[colName] === undefined || r[colName] === null || r[colName] === '').length;
    
    if (nullCount > 0) {
      report.missingHandled[colName] = options.missingStrategy;
      
      if (options.missingStrategy === 'drop') {
        data = data.filter(r => r[colName] !== undefined && r[colName] !== null && r[colName] !== '');
      } else {
        // Compute surrogate
        const validValues = data
          .map(r => r[colName])
          .filter(v => v !== undefined && v !== null && v !== '');
        
        let replacement: any = 0;
        
        if (col.type === 'numeric') {
          const numValues = validValues.map(Number);
          if (options.missingStrategy === 'mean') {
            const sum = numValues.reduce((a, b) => a + b, 0);
            replacement = numValues.length > 0 ? sum / numValues.length : 0;
          } else if (options.missingStrategy === 'median') {
            const sorted = [...numValues].sort((a, b) => a - b);
            replacement = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0;
          } else { // mode
            replacement = getMode(numValues);
          }
        } else {
          // Categorical/boolean mode
          replacement = getMode(validValues);
        }
        
        // Apply replacement
        for (const row of data) {
          if (row[colName] === undefined || row[colName] === null || row[colName] === '') {
            row[colName] = replacement;
          }
        }
      }
    }
  }

  // 3. Outlier Detection (for numeric columns except target)
  let outlierRowsToDrop = new Set<number>();
  for (const col of columns) {
    const colName = col.name;
    if (col.type === 'numeric' && colName !== targetColumn) {
      const vals = data.map(r => Number(r[colName])).filter(v => !isNaN(v));
      if (vals.length > 4) {
        const sorted = [...vals].sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        vals.forEach((v, idx) => {
          if (v < lowerBound || v > upperBound) {
            report.outliersDetected++;
            if (options.handleOutliers === 'clip') {
              const clipped = Math.max(lowerBound, Math.min(upperBound, v));
              data[idx][colName] = clipped;
            } else if (options.handleOutliers === 'remove') {
              outlierRowsToDrop.add(idx);
            }
          }
        });
      }
    }
  }

  if (options.handleOutliers === 'remove' && outlierRowsToDrop.size > 0) {
    data = data.filter((_, idx) => !outlierRowsToDrop.has(idx));
  }

  // 4. Label / Binary Encoding for String/Boolean columns
  for (const col of columns) {
    const colName = col.name;
    if (col.type === 'categorical' || col.type === 'boolean') {
      const uniqueVals = Array.from(new Set(data.map(r => r[colName])));
      const map: Record<string | number, number> = {};
      uniqueVals.forEach((val, idx) => {
        map[val] = idx;
      });
      labelEncodings[colName] = map;
      report.encodedColumns.push(colName);
      
      // Update values in-place
      for (const row of data) {
        row[colName] = map[row[colName]] ?? 0;
      }
    }
  }

  // 5. Feature Scaling
  if (options.scalingMethod !== 'none') {
    for (const col of columns) {
      const colName = col.name;
      // Scale numerical independent variables, optionally skip target
      if (col.type === 'numeric' && colName !== targetColumn) {
        const vals = data.map(r => Number(r[colName]));
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        const sum = vals.reduce((a, b) => a + b, 0);
        const mean = sum / vals.length;
        const sqDiffSum = vals.reduce((a, b) => a + (b - mean) ** 2, 0);
        const std = Math.sqrt(sqDiffSum / vals.length) || 1;
        
        scalingParams[colName] = { mean, std, min, max };
        report.scaledColumns.push(colName);

        for (const row of data) {
          const originalValue = Number(row[colName]);
          if (options.scalingMethod === 'standard') {
            row[colName] = (originalValue - mean) / std;
          } else if (options.scalingMethod === 'minmax') {
            row[colName] = max !== min ? (originalValue - min) / (max - min) : 0;
          }
        }
      }
    }
  }

  // 6. Feature Selection (We can prioritize features based on standard mathematical correlations with target)
  const numericColumns = columns.filter(c => c.type === 'numeric' || labelEncodings[c.name] !== undefined);
  let finalFeatures = numericColumns.map(c => c.name);
  
  if (targetColumn && finalFeatures.includes(targetColumn)) {
    finalFeatures = finalFeatures.filter(f => f !== targetColumn);
  }

  if (targetColumn && options.featureSelectionCount > 0 && finalFeatures.length > options.featureSelectionCount) {
    // Select features that have absolute Spearman/Pearson-like correlation with target
    const targetVals = data.map(r => Number(r[targetColumn]));
    const targetMean = targetVals.reduce((a, b) => a + b, 0) / targetVals.length;
    
    const correlations = finalFeatures.map(col => {
      const vals = data.map(r => Number(r[col]));
      const valMean = vals.reduce((a, b) => a + b, 0) / vals.length;
      
      let num = 0;
      let denY = 0;
      let denX = 0;
      
      for (let i = 0; i < data.length; i++) {
        const dx = vals[i] - valMean;
        const dy = targetVals[i] - targetMean;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      
      const corr = denX * denY > 0 ? num / Math.sqrt(denX * denY) : 0;
      return { col, score: Math.abs(corr) };
    });

    correlations.sort((a, b) => b.score - a.score);
    finalFeatures = correlations.slice(0, options.featureSelectionCount).map(c => c.col);
  }
  
  report.selectedFeatures = finalFeatures;

  // Build column descriptions
  const processedDataNumeric = data.map(row => {
    const newRow: Record<string, number> = {};
    Object.keys(row).forEach(k => {
      newRow[k] = Number(row[k]) || 0;
    });
    return newRow;
  });

  // Re-create columns with updated stats
  const processedColumns = columns.map(c => {
    const colName = c.name;
    const values = processedDataNumeric.map(r => r[colName]);
    const info: ColumnInfo = {
      name: colName,
      type: 'numeric', // Everything is numeric in processed dataset
      missingValues: 0,
      uniqueValues: new Set(values).size,
      min: Math.min(...values),
      max: Math.max(...values),
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      median: [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]
    };
    return info;
  });

  return {
    name: dataset.name,
    filename: dataset.filename,
    columns: processedColumns,
    rawData: dataset.rawData, // preserve raw
    processedData: processedDataNumeric, // processed numerical rows
    labelEncodings,
    scalingParams,
    rowCount: processedDataNumeric.length,
    colCount: processedColumns.length,
    report
  };
}

function getMode(arr: any[]): any {
  if (arr.length === 0) return null;
  const count: Record<string, number> = {};
  let maxCount = 0;
  let mode = arr[0];
  
  for (const item of arr) {
    const key = String(item);
    count[key] = (count[key] || 0) + 1;
    if (count[key] > maxCount) {
      maxCount = count[key];
      mode = item;
    }
  }
  return mode;
}
