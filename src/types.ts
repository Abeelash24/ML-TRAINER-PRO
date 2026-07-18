/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type DataType = 'numeric' | 'categorical' | 'boolean' | 'date';

export interface ColumnInfo {
  name: string;
  type: DataType;
  missingValues: number;
  duplicateCount?: number;
  uniqueValues: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  stdDev?: number;
}

export interface Dataset {
  name: string;
  filename?: string;
  rawData: Record<string, any>[];
  columns: ColumnInfo[];
  rowCount: number;
  colCount: number;
}

export interface PreprocessingReport {
  missingHandled: Record<string, string>; // column -> strategy used
  duplicatesRemoved: number;
  encodedColumns: string[]; // columns that were encoded
  scaledColumns: string[]; // columns that were standardized/scaled
  outliersDetected: number;
  selectedFeatures: string[];
}

export interface PreprocessedDataset extends Dataset {
  processedData: Record<string, number>[]; // numerical inputs ready for model feeding
  labelEncodings: Record<string, Record<string | number, number>>; // column -> original -> numeric mapping
  scalingParams?: Record<string, { mean: number; std: number; min: number; max: number }>;
  report: PreprocessingReport;
}

export type ProblemType = 'classification' | 'regression';

export interface ModelMetrics {
  trainingAccuracy?: number; // classification
  testingAccuracy?: number; // classification
  precision?: number;
  recall?: number;
  f1Score?: number;
  rocAuc?: number;
  
  mae?: number; // regression
  mse?: number; // regression
  rmse?: number; // regression
  r2Score?: number; // regression
  
  crossValScores: number[];
  cvMean: number;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][]; // actual × predicted
}

export interface TrainedModel {
  id: string; // 'rf' | 'dt' | 'linear_log'
  name: string;
  problemType: ProblemType;
  metrics: ModelMetrics;
  featureImportances: Record<string, number>;
  confusionMatrix?: ConfusionMatrix;
  hyperparameters: Record<string, any>;
  rawCoefficients?: Record<string, number>; // for regression/logistic representation
  intercept?: number;
  modelCode: string; // Python code representation
}

export interface ModelComparison {
  models: Record<string, TrainedModel>;
  bestModelId: string;
  ranking: string[];
}
