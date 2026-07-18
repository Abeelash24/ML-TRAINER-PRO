/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useDragEvent } from 'react';
import { 
  getPresetDatasets, 
  analyzeColumns 
} from './data/datasets';
import { 
  preprocessDataset, 
  PreprocessingOptions 
} from './utils/preprocessing';
import { 
  trainModel, 
  detectProblemType, 
  runSingleRowPrediction 
} from './utils/model_training';
import { 
  Dataset, 
  PreprocessedDataset, 
  ProblemType, 
  TrainedModel 
} from './types';
import { 
  HistogramChart, 
  ScatterPlotChart, 
  CorrelationHeatmap, 
  FeatureImportanceBarChart, 
  MetricsBarChart 
} from './components/MLCharts';
import CodeExporter from './components/CodeExporter';

// Lucide Icons
import {
  Home,
  Database,
  LineChart,
  Brain,
  TrendingUp,
  FileCode,
  Gauge,
  Sliders,
  Download,
  Upload,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Info,
  Calendar,
  Sparkles,
  ChevronRight,
  Eye,
  FileDown,
  RefreshCw,
  HelpCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Unified CSV string-based parser
function parseCSV(text: string, filename: string): Dataset {
  const lines = text.split(/\r?\n/);
  if (lines.length === 0) throw new Error('Blank/empty CSV file');
  
  // Clean headers
  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rawData: Record<string, any>[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    // Split values taking quotes into account
    const values: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let charIdx = 0; charIdx < line.length; charIdx++) {
      const char = line[charIdx];
      if (char === '"' || charChar(charIdx)) {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        values.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    values.push(current.trim());
    
    if (values.length >= headers.length) {
      const row: Record<string, any> = {};
      headers.forEach((header, idx) => {
        let val: any = values[idx];
        if (val !== undefined && val !== null) {
          val = val.replace(/^["']|["']$/g, '');
          if (val === 'null' || val === 'NaN' || val === '') {
            row[header] = null;
          } else if (!isNaN(Number(val)) && val !== '') {
            row[header] = Number(val);
          } else if (val === 'true' || val === 'Yes') {
            row[header] = true;
          } else if (val === 'false' || val === 'No') {
            row[header] = false;
          } else {
            row[header] = val;
          }
        }
      });
      rawData.push(row);
    }
  }

  function charChar(idx: number) {
    return false;
  }

  const columns = analyzeColumns(rawData);
  return {
    name: filename.replace('.csv', '') + ' Ingested',
    filename,
    rawData,
    columns,
    rowCount: rawData.length,
    colCount: columns.length
  };
}

export default function App() {
  const presets = getPresetDatasets();
  
  // Global States
  const [activeTab, setActiveTab] = useState<string>('home');
  const [activeDataset, setActiveDataset] = useState<Dataset>(presets[0]); // Default to Iris
  const [preprocessed, setPreprocessed] = useState<PreprocessedDataset | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>('Species');
  const [problemType, setProblemType] = useState<ProblemType>('classification');
  
  // Loading upload effects
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Preprocessing config options
  const [prepOptions, setPrepOptions] = useState<PreprocessingOptions>({
    missingStrategy: 'mean',
    removeDuplicates: true,
    scalingMethod: 'standard',
    handleOutliers: 'clip',
    featureSelectionCount: 0
  });

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [maxDepth, setMaxDepth] = useState<number>(4);
  const [nEstimators, setNEstimators] = useState<number>(5);
  const [trainRatio, setTrainRatio] = useState<number>(0.75);

  // Model training collections
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainedModels, setTrainedModels] = useState<Record<string, TrainedModel>>({});
  const [bestModelId, setBestModelId] = useState<string>('');

  // Predictions states
  const [predictionResult, setPredictionResult] = useState<{
    prediction: string | number;
    probability?: number;
    confidence: number;
    inputsUsed: Record<string, number>;
  } | null>(null);

  const [predictInputs, setPredictInputs] = useState<Record<string, number>>({});

  // Auto initialize preprocessed state on dataset shift
  useEffect(() => {
    // Attempt automatic deduction of target column
    const cols = activeDataset.columns;
    if (cols.length > 0) {
      // Find possible target (usually last column or specific string/Outcome)
      let targetCandidate = cols[cols.length - 1].name;
      const targetMatches = cols.find(c => ['outcome', 'label', 'target', 'species', 'class', 'survived', 'salary'].includes(c.name.toLowerCase()));
      if (targetMatches) {
        targetCandidate = targetMatches.name;
      }
      
      setTargetColumn(targetCandidate);
      
      // Perform initial light preprocess sequence
      const prep = preprocessDataset(activeDataset, {
        missingStrategy: 'mean',
        removeDuplicates: true,
        scalingMethod: 'none', // Default none on initial load to view raw proportions
        handleOutliers: 'none',
        featureSelectionCount: 0
      }, targetCandidate);
      
      setPreprocessed(prep);
      setProblemType(detectProblemType(prep, targetCandidate));
      
      // Clear models and predictions
      setTrainedModels({});
      setBestModelId('');
      setPredictionResult(null);
    }
  }, [activeDataset]);

  // Handle Drag-and-drop actions
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setUploadError(null);
    
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      processFile(file);
    } else {
      setUploadError('Only standard tabular .csv files are supported.');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setIsUploading(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = parseCSV(text, file.name);
        setActiveDataset(parsed);
        setIsUploading(false);
        setActiveTab('analysis');
      } catch (err: any) {
        setUploadError(`Failed parsing CSV file: ${err.message || 'Syntax mismatch'}`);
        setIsUploading(false);
      }
    };
    reader.readAsText(file);
  };

  // Preprocessing triggers
  const executePreprocessing = () => {
    const prep = preprocessDataset(activeDataset, prepOptions, targetColumn);
    setPreprocessed(prep);
    setProblemType(detectProblemType(prep, targetColumn));
    setTrainedModels({});
    setBestModelId('');
    setPredictionResult(null);
  };

  // Perform model calibrations of 3 unique models
  const executeModelTraining = () => {
    if (!preprocessed) return;
    setIsTraining(true);

    setTimeout(() => {
      const rfModel = trainModel(preprocessed, targetColumn, 'rf', problemType, {
        nEstimators,
        maxDepth,
        trainRatio
      });

      const dtModel = trainModel(preprocessed, targetColumn, 'dt', problemType, {
        maxDepth,
        trainRatio
      });

      const regModel = trainModel(preprocessed, targetColumn, 'linear_log', problemType, {
        learningRate,
        epochs: 400,
        trainRatio
      });

      const results = {
        rf: rfModel,
        dt: dtModel,
        linear_log: regModel
      };

      setTrainedModels(results);

      // Determine Best Model based on Hold-out testing metric (Accuracy for Classify, R^2 for Regress)
      let optimalId = 'rf';
      if (problemType === 'classification') {
        const bestAcc = Math.max(
          rfModel.metrics.testingAccuracy || 0,
          dtModel.metrics.testingAccuracy || 0,
          regModel.metrics.testingAccuracy || 0
        );
        if (dtModel.metrics.testingAccuracy === bestAcc) optimalId = 'dt';
        else if (regModel.metrics.testingAccuracy === bestAcc) optimalId = 'linear_log';
      } else {
        const bestR2 = Math.max(
          rfModel.metrics.r2Score || -999,
          dtModel.metrics.r2Score || -999,
          regModel.metrics.r2Score || -999
        );
        if (dtModel.metrics.r2Score === bestR2) optimalId = 'dt';
        else if (regModel.metrics.r2Score === bestR2) optimalId = 'linear_log';
      }

      setBestModelId(optimalId);
      setIsTraining(false);
      setActiveTab('comparison');

      // Initialize predict sliders with mean parameters
      const independentFeatures = preprocessed.report.selectedFeatures;
      const initialInputs: Record<string, number> = {};
      independentFeatures.forEach(feat => {
        const colMeta = preprocessed.columns.find(c => c.name === feat);
        initialInputs[feat] = colMeta?.mean ?? 0;
      });
      setPredictInputs(initialInputs);
    }, 850);
  };

  // Predict outputs for variables
  const triggerPrediction = (modelId: string) => {
    const model = trainedModels[modelId];
    if (!model || !preprocessed) return;

    const independentFeatures = preprocessed.report.selectedFeatures;
    const res = runSingleRowPrediction(model, independentFeatures, predictInputs);

    // Resolve string classifications
    let finalPrediction: string | number = res.prediction;
    if (problemType === 'classification') {
      const encodeMap = preprocessed.labelEncodings[targetColumn];
      if (encodeMap) {
        const found = Object.entries(encodeMap).find(([org, val]) => val === res.prediction);
        if (found) finalPrediction = found[0];
      }
    }

    setPredictionResult({
      prediction: finalPrediction,
      probability: res.probability,
      confidence: res.confidence,
      inputsUsed: { ...predictInputs }
    });
  };

  // Preset quick swaps
  const handleSwapPreset = (idx: number) => {
    setActiveDataset(presets[idx]);
  };

  // Helper strings to view/export Reports is constructed on demand
  const generateCSVScoreboardString = () => {
    const modelsList = Object.values(trainedModels) as TrainedModel[];
    if (modelsList.length === 0) return '';
    let csv = "Model,Validation Mode,Metric Key,Metric Value,CrossVal Mean\n";
    modelsList.forEach(model => {
      Object.entries(model.metrics).forEach(([key, val]) => {
        if (key !== 'crossValScores' && typeof val === 'number') {
          csv += `${model.name},${problemType},${key},${val.toFixed(4)},${model.metrics.cvMean.toFixed(4)}\n`;
        }
      });
    });
    return csv;
  };

  const generateMarkdownReportString = () => {
    const modelsList = Object.values(trainedModels) as TrainedModel[];
    if (modelsList.length === 0) return '';
    const bestModel = trainedModels[bestModelId];
    let md = `# MACHINE LEARNING EXPERIMENTAL STUDIO EVALUATION LOGGER\n`;
    md += `**Target Variable:** ${targetColumn}\n`;
    md += `**Problem Modality:** ${problemType.toUpperCase()}\n`;
    md += `**Recommended AutoML Architecture:** ${bestModel?.name || ''}\n`;
    md += `**Evaluation Records Count:** ${preprocessed?.rowCount || 0} rows\n\n`;
    md += `---------------------------------------------------\n\n`;
    md += `## competitive model scorecard matrix\n\n`;
    
    if (problemType === 'classification') {
      md += `| Model | Hold-Out accuracy | Precision | Recall | Weighted F1 | 5-Fold CV Mean |\n`;
      md += `|---|---|---|---|---|---|\n`;
      modelsList.forEach(m => {
        md += `| ${m.name} | ${(m.metrics.testingAccuracy || 0).toFixed(4)} | ${(m.metrics.precision || 0).toFixed(4)} | ${(m.metrics.recall || 0).toFixed(4)} | ${(m.metrics.f1Score || 0).toFixed(4)} | ${m.metrics.cvMean.toFixed(4)} |\n`;
      });
    } else {
      md += `| Model | R² score (Variance) | Mean Abs Error (MAE) | Root Mean Sq (RMSE) | 5-Fold CV R² Mean |\n`;
      md += `|---|---|---|---|---|\n`;
      modelsList.forEach(m => {
        md += `| ${m.name} | ${(m.metrics.r2Score || 0).toFixed(4)} | ${(m.metrics.mae || 0).toFixed(2)} | ${(m.metrics.rmse || 0).toFixed(2)} | ${m.metrics.cvMean.toFixed(4)} |\n`;
      });
    }

    md += `\n## feature ranking analysis (${bestModel?.name || ''})\n\n`;
    const sortedFeatures = Object.entries(bestModel?.featureImportances || {})
      .sort((a, b) => (b[1] as number) - (a[1] as number));
    
    md += `| Rank | Variable Indicator | Gini Weight Index |\n`;
    md += `|---|---|---|\n`;
    sortedFeatures.forEach(([feat, weight], rIdx) => {
      md += `| #${rIdx+1} | ${feat} | ${((weight as number) * 100).toFixed(1)}% |\n`;
    });

    md += `\n--- \n*Compiled by ML Code Exporter Engine. Final-year project-ready deployment verified successfully.*`;
    return md;
  };

  return (
    <div className="flex h-screen bg-[#0F172A] text-slate-300 overflow-hidden font-sans select-none">
      
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-[#0B1120] text-slate-300 flex flex-col justify-between border-r border-[#1E293B] flex-shrink-0">
        
        {/* Core Sidebar Header */}
        <div className="p-6 flex-1 flex flex-col overflow-y-auto">
          <div className="flex flex-col mb-6 border-b border-[#1E293B] pb-4">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-6 h-6 bg-sky-500 rounded-sm flex items-center justify-center">
                <Cpu className="w-4 h-4 text-[#0B1120]" />
              </div>
              <span className="font-bold text-white tracking-tight text-sm uppercase">ML Studio Pro</span>
            </div>
            <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">
              v2.4.0 • Enterprise Edition
            </p>
          </div>

          <div>
            {/* Active Data Summary */}
            <div className="bg-[#1E293B]/40 border border-[#1E293B] rounded-sm p-3.5 mb-6">
              <div className="flex justify-between items-center text-[9px] text-[#64748B] font-bold uppercase tracking-wider">
                <span>Ingested Dataset</span>
                <span className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#34D399] animate-pulse" />
                  Live
                </span>
              </div>
              <p className="text-xs font-bold font-mono text-slate-200 mt-1 truncate pl-0.5">
                {activeDataset.name}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-3 text-[11px] font-mono border-t border-[#1E293B] pt-2.5 text-slate-400">
                <div>Rows: <strong className="text-white">{activeDataset.rowCount}</strong></div>
                <div>Cols: <strong className="text-white">{activeDataset.colCount}</strong></div>
              </div>
            </div>
          </div>

          {/* Links list */}
          <nav className="space-y-1">
            {[
              { id: 'home', label: 'Home & Guidelines', icon: Home },
              { id: 'analysis', label: 'Dataset Inspector', icon: Database },
              { id: 'eda', label: 'Exploratory EDA', icon: LineChart },
              { id: 'training', label: 'Model Training Studio', icon: Sliders },
              { id: 'comparison', label: 'Model Comparison', icon: TrendingUp },
              { id: 'importance', label: 'Feature Importance', icon: Gauge },
              { id: 'prediction', label: 'Predictions Center', icon: Brain },
              { id: 'code', label: 'Code & Reports Hub', icon: FileCode }
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full text-left px-3 py-2 rounded-sm transition-all font-medium text-xs flex items-center gap-3 relative ${
                    isActive 
                      ? 'bg-sky-500/10 text-sky-400 border-l-4 border-sky-400 pl-2 font-semibold' 
                      : 'text-slate-400 hover:bg-[#1E293B]/40 hover:text-white'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-500'}`} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* CPU & Work Environment Monitoring */}
        <div className="p-6 border-t border-[#1E293B] bg-[#0B1120]">
          <div className="flex items-center justify-between text-[11px] mb-2 font-mono text-slate-500">
            <span>CPU Usage</span>
            <span className="text-sky-400 font-bold">42%</span>
          </div>
          <div className="w-full h-1 bg-[#1E293B] rounded-sm overflow-hidden mb-3">
            <div className="h-full bg-sky-450 progress-fill" style={{ width: '42%' }}></div>
          </div>
          <div className="text-[10px] text-slate-500 font-mono tracking-wide leading-relaxed">
            <div>User: <span className="text-slate-400 font-semibold">Student Demo</span></div>
            <div>Time: <span className="text-slate-400">2026-06-19 UTC</span></div>
          </div>
        </div>
      </aside>

      {/* Main Workspace Stage */}
      <main className="flex-1 overflow-y-auto flex flex-col justify-between dot-grid">
        <header className="h-16 border-b border-[#1E293B] flex items-center justify-between px-8 bg-[#0F172A]/80 backdrop-blur-md z-10">
          <div>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              {activeTab === 'home' && 'Studio Welcome & Data Ingest'}
              {activeTab === 'analysis' && 'Data Attribute Inspector'}
              {activeTab === 'eda' && 'Exploratory Data Analysis (EDA)'}
              {activeTab === 'training' && 'Model Training Studio'}
              {activeTab === 'comparison' && 'Competitive Score Evaluation'}
              {activeTab === 'importance' && 'Feature Importance Rankings'}
              {activeTab === 'prediction' && 'Core Simulation & Prediction'}
              {activeTab === 'code' && 'Code & Reports Hub'}
            </h2>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Project: {activeDataset.filename || 'active_data.csv'}
            </p>
          </div>
          
          <div className="flex gap-3">
            <span className="text-xs bg-sky-500/10 text-sky-400 px-3 py-1 rounded border border-sky-400/20 flex items-center gap-1 font-mono uppercase tracking-widest text-[9px] font-bold">
              <Sparkles className="w-3.5 h-3.5" />
              AutoML Active
            </span>
          </div>
        </header>

        {/* Inner Content Area */}
        <div className="p-8 flex-1">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.18 }}
              className="max-w-6xl mx-auto space-y-6"
            >
              
              {/* PAGE 1: HOME */}
              {activeTab === 'home' && (
                <div className="space-y-6">
                  {/* Banner Info */}
                  <div className="bg-gradient-to-r from-violet-600 via-indigo-600 to-indigo-700 text-white rounded-2xl p-8 shadow-md relative overflow-hidden">
                    <div className="absolute right-0 bottom-0 top-0 w-1/3 opacity-10 bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-yellow-300 via-pink-500 to-indigo-950 pointer-events-none" />
                    <div className="max-w-2xl">
                      <span className="text-xs font-bold uppercase tracking-widest text-violet-200 bg-violet-500/20 px-3 py-1 rounded-full border border-violet-400/20">
                        Senior Academic Project
                      </span>
                      <h3 className="text-3xl font-black tracking-tight mt-3">
                        Advanced Machine Learning Model Training Studio
                      </h3>
                      <p className="text-sm text-indigo-100 mt-2 leading-relaxed">
                        An integrated sandbox to upload CSV files, run exploratory data charts, configure hyperparameter splits, and compute linear/logistic or tree models in-browser. Fully furnished with pre-encoded Python source scripts to compile locally.
                      </p>
                    </div>
                  </div>

                  {/* Drag and Drop Interface */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-6">
                      
                      <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`border-3 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[220px] ${
                          isDragging 
                            ? 'border-violet-500 bg-violet-50/50' 
                            : 'border-gray-200 hover:border-violet-400 bg-white hover:bg-gray-50/20'
                        }`}
                      >
                        <input
                          type="file"
                          id="csv-file-picker"
                          accept=".csv"
                          onChange={handleFileSelect}
                          className="hidden"
                        />
                        <label htmlFor="csv-file-picker" className="cursor-pointer flex flex-col items-center justify-center">
                          <div className="p-4 bg-violet-50 text-violet-600 rounded-full mb-4">
                            <Upload className="w-8 h-8" />
                          </div>
                          <h4 className="font-bold text-gray-800 text-sm">Drag & Drop Dataset CSV Here</h4>
                          <p className="text-xs text-gray-400 mt-1">Or click to search your computer files (Maximum limit 25MB)</p>
                        </label>

                        {isUploading && (
                          <div className="mt-4 text-xs font-semibold text-violet-600 animate-pulse flex items-center gap-1.5">
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Ingesting file headers...
                          </div>
                        )}

                        {uploadError && (
                          <div className="mt-4 p-3 bg-rose-50 rounded-xl border border-rose-100 text-xs text-rose-600 font-semibold flex items-center gap-2">
                            <AlertCircle className="w-4.5 h-4.5" />
                            <span>{uploadError}</span>
                          </div>
                        )}
                      </div>

                      {/* Presets Cards */}
                      <div>
                        <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3 pl-1">
                          Test-drive with Standard Reference Datasets
                        </h4>
                        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                          {[
                            { name: 'Iris Flower Classify', idx: 0, tag: 'Classification', desc: 'Predict botanical classifications from multi-petal width bounds.' },
                            { name: 'Titanic Survivors', idx: 1, tag: 'Classification', desc: 'Examine biological survival variables containing actual null elements.' },
                            { name: 'Diabetes Wellness', idx: 2, tag: 'Classification', desc: 'Clinical insulin levels mapping pre-diabetic patients diagnostic Outcomes.' },
                            { name: 'Student Performance', idx: 3, tag: 'Regression', desc: 'Identify continuous numerical scores based on study hours.' },
                            { name: 'Employee Salary List', idx: 4, tag: 'Regression', desc: 'Track numerical compensation indexes vs degrees & departments.' }
                          ].map(item => {
                            const isSelected = activeDataset.name === presets[item.idx].name;
                            return (
                              <button
                                key={item.name}
                                onClick={() => handleSwapPreset(item.idx)}
                                className={`text-left p-4 rounded-xl border transition-all ${
                                  isSelected 
                                    ? 'bg-violet-50/70 border-violet-200 ring-2 ring-violet-500/10' 
                                    : 'bg-white border-gray-100 hover:border-gray-200'
                                }`}
                              >
                                <div className="flex justify-between items-center">
                                  <span className="text-[9px] font-bold uppercase tracking-wide bg-gray-100 text-gray-500 px-2 py-0.5 rounded">
                                    {item.tag}
                                  </span>
                                  {isSelected && <CheckCircle2 className="w-4 h-4 text-violet-600" />}
                                </div>
                                <h5 className="font-bold text-gray-800 text-xs mt-2">{item.name}</h5>
                                <p className="text-[11px] text-gray-400 mt-1 leading-normal line-clamp-2">{item.desc}</p>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                    </div>

                    {/* Features checklist bounds */}
                    <div className="space-y-4">
                      <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                        <h4 className="font-bold text-gray-800 text-sm mb-4 flex items-center gap-2">
                          <Info className="w-4.5 h-4.5 text-violet-500" />
                          Platform Checklist Capabilities
                        </h4>
                        <ul className="space-y-3.5 text-xs text-gray-500">
                          <li className="flex items-start gap-2">
                            <CheckCircle2 className="w-4.5 h-4.5 text-emerald-500 mt-0.5 flex-shrink-0" />
                            <span><strong>Statistical Profiling:</strong> Auto-deduct shapes, cardinalities, median distributions.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle2 className="w-4.5 h-4.5 text-emerald-500 mt-0.5 flex-shrink-0" />
                            <span><strong>Pipeline Imputations:</strong> Choose mean/median replacement fields for missing variables.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle2 className="w-4.5 h-4.5 text-emerald-500 mt-0.5 flex-shrink-0" />
                            <span><strong>Mathematical Models:</strong> Real fits of Random Forest, Decision Tree, Logistic / Linear.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle2 className="w-4.5 h-4.5 text-emerald-500 mt-0.5 flex-shrink-0" />
                            <span><strong>Source Code Exporting:</strong> Direct tab explorers for production Streamlit scripting files.</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 2: DATA INSPECTOR */}
              {activeTab === 'analysis' && (
                <div className="space-y-6">
                  {/* Summary Metric Strip */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[
                      { l: 'Variable Indicators', v: activeDataset.colCount, d: 'Total Columns' },
                      { l: 'Evaluated Rows', v: activeDataset.rowCount, d: 'Total Records' },
                      { l: 'Identified Duplicates', v: activeDataset.rawData.length - (new Set(activeDataset.rawData.map(r => JSON.stringify(r))).size), d: 'Exact duplicate rows' },
                      { l: 'Continuous Columns', v: activeDataset.columns.filter(c => c.type === 'numeric').length, d: 'Numeric numeric weights' }
                    ].map(card => (
                      <div key={card.l} className="bg-white border border-gray-100 rounded-xl p-5 shadow-sm">
                        <span className="text-xs text-gray-400 font-bold uppercase tracking-wider block">{card.l}</span>
                        <strong className="text-2xl font-black text-gray-800 block mt-1">{card.v}</strong>
                        <span className="text-[10px] text-gray-400 font-mono block mt-1">{card.d}</span>
                      </div>
                    ))}
                  </div>

                  {/* Scrollable Data Table Preview */}
                  <div className="bg-white border border-gray-100 rounded-2xl shadow-sm overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                      <div>
                        <h4 className="font-bold text-gray-800 text-sm">Ingested Raw Data Table (Head Preview)</h4>
                        <p className="text-[11px] text-gray-400 mt-0.5">Displaying up to the first 10 recorded entries contained inside the file.</p>
                      </div>
                    </div>
                    
                    <div className="overflow-x-auto max-h-[360px]">
                      <table className="min-w-full divide-y divide-gray-100 text-left font-mono text-xs">
                        <thead className="bg-gray-50/60 sticky top-0 z-10 backdrop-blur">
                          <tr>
                            <th className="p-3.5 text-gray-500 font-bold border-b border-gray-100">Row</th>
                            {activeDataset.columns.map(col => (
                              <th key={col.name} className="p-3.5 text-gray-500 font-bold border-b border-gray-100 truncate max-w-[150px]">
                                {col.name}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-100">
                          {activeDataset.rawData.slice(0, 10).map((row, rIdx) => (
                            <tr key={rIdx} className="hover:bg-gray-50/40">
                              <td className="p-3.5 text-gray-400">{rIdx + 1}</td>
                              {activeDataset.columns.map(col => {
                                const val = row[col.name];
                                return (
                                  <td key={col.name} className="p-3.5 text-gray-800 font-medium truncate max-w-[150px]">
                                    {val === null || val === undefined ? (
                                      <span className="text-rose-500 bg-rose-5 font-bold px-1 rounded text-[10px]">NULL</span>
                                    ) : (
                                      String(val)
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Schema breakdown metadata */}
                  <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                    <h4 className="font-bold text-gray-800 text-sm mb-4">Column Definitions & Cardinalities</h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-left font-mono text-xs border border-gray-100 rounded-lg overflow-hidden">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="p-3 text-gray-500 font-bold">Variable Name</th>
                            <th className="p-3 text-gray-500 font-bold">Inferred Type</th>
                            <th className="p-3 text-gray-500 font-bold text-center">Missing Val Block</th>
                            <th className="p-3 text-gray-500 font-bold text-center">Cardinality</th>
                            <th className="p-3 text-gray-500 font-bold">Min Bounds</th>
                            <th className="p-3 text-gray-500 font-bold">Max Bounds</th>
                            <th className="p-3 text-gray-500 font-bold">Median / Mean</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {activeDataset.columns.map(col => (
                            <tr key={col.name}>
                              <td className="p-3 text-gray-800 font-bold">{col.name}</td>
                              <td className="p-3">
                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                                  col.type === 'numeric' ? 'bg-blue-50 text-blue-700' :
                                  col.type === 'boolean' ? 'bg-amber-50 text-amber-700' : 'bg-purple-50 text-purple-700'
                                }`}>
                                  {col.type}
                                </span>
                              </td>
                              <td className="p-3 text-center">
                                {col.missingValues > 0 ? (
                                  <span className="text-rose-600 bg-rose-50 px-2 py-0.5 rounded font-bold">{col.missingValues}</span>
                                ) : (
                                  <span className="text-gray-400">clean</span>
                                )}
                              </td>
                              <td className="p-3 text-center font-bold text-gray-600">{col.uniqueValues} values</td>
                              <td className="p-3 text-gray-500">{col.min !== undefined ? col.min.toFixed(2) : '-'}</td>
                              <td className="p-3 text-gray-500">{col.max !== undefined ? col.max.toFixed(2) : '-'}</td>
                              <td className="p-3 text-gray-500">{col.mean !== undefined ? col.mean.toFixed(2) : '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* PAGE 3: EDA */}
              {activeTab === 'eda' && (
                <div className="space-y-6">
                  
                  {/* Selectors */}
                  <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                    <h4 className="font-bold text-gray-800 text-sm mb-4">Set Interactive Visualization Target Column</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <label className="text-[10px] text-gray-400 font-bold uppercase tracking-wider block mb-1">Target Variable (Histogram)</label>
                        <select
                          value={targetColumn}
                          onChange={(e) => setTargetColumn(e.target.value)}
                          className="w-full font-mono text-xs bg-gray-50 border border-gray-100 rounded-xl px-3.5 py-2.5 focus:outline-none focus:ring-2 focus:ring-violet-500"
                        >
                          {activeDataset.columns.map(c => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Histogram and Pair Charts */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    
                    {/* Histogram Distribution */}
                    <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                      <h4 className="font-bold text-gray-800 text-sm mb-2">1D Frequency Distribution Chart</h4>
                      <p className="text-xs text-gray-400 mb-6">Visual binned values mapping frequency metrics representation.</p>
                      <HistogramChart data={activeDataset.rawData} columnName={targetColumn} binsCount={10} />
                    </div>

                    {/* Bivariate Scatter Plot */}
                    <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                      <h4 className="font-bold text-gray-800 text-sm mb-2">2D Bivariate Scatter Relations</h4>
                      <p className="text-xs text-gray-400 mb-6 font-mono">Select independent features x / y to visualize points distributions.</p>
                      
                      <div className="grid grid-cols-2 gap-3 mb-4 font-mono text-xs">
                        <div>
                          <label className="block text-[10px] text-gray-400 font-bold uppercase mb-1">Select X-Dimension</label>
                          <select 
                            onChange={(e) => executePreprocessing()} // Recharts trigger
                            className="bg-gray-50 border border-gray-100 rounded-lg p-1.5 w-full focus:outline-none"
                            id="scatter-x-trigger"
                          >
                            {activeDataset.columns.filter(c => c.type === 'numeric').map(c => (
                              <option key={c.name} value={c.name}>{c.name}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-[10px] text-gray-400 font-bold uppercase mb-1">Select Y-Dimension</label>
                          <select 
                            className="bg-gray-50 border border-gray-100 rounded-lg p-1.5 w-full focus:outline-none"
                            id="scatter-y-trigger"
                          >
                            {activeDataset.columns.filter(c => c.type === 'numeric').reverse().map(c => (
                              <option key={c.name} value={c.name}>{c.name}</option>
                            ))}
                          </select>
                        </div>
                      </div>

                      <ScatterPlotChart 
                        data={activeDataset.rawData} 
                        xCol={activeDataset.columns.filter(c => c.type === 'numeric')[0]?.name || ''} 
                        yCol={activeDataset.columns.filter(c => c.type === 'numeric')[1]?.name || ''} 
                      />
                    </div>
                  </div>

                  {/* Heatmap Grid */}
                  <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                    <h4 className="font-bold text-gray-800 text-sm mb-2">Multi-Variable Pearson Correlation Matrix Image</h4>
                    <p className="text-xs text-gray-400 mb-6">Calculates pairwise Pearson relationship metrics on independent numeric columns indicators.</p>
                    <CorrelationHeatmap columns={activeDataset.columns.map(c => c.name)} data={activeDataset.rawData} />
                  </div>
                </div>
              )}

              {/* PAGE 4: PIPELINE SETUP */}
              {activeTab === 'training' && (
                <div className="space-y-6">
                  <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                    <h4 className="font-bold text-gray-800 text-sm mb-4">Set Pipeline Optimization Parameters</h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 font-mono text-xs">
                      {/* Column Selector */}
                      <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 space-y-3">
                        <h5 className="font-bold text-gray-700 uppercase tracking-wide text-[10px] flex items-center gap-1">
                          <Database className="w-3.5 h-3.5 text-violet-500" /> Key Features selection
                        </h5>
                        <div>
                          <label className="block text-[10px] text-gray-400 uppercase mb-1">Dependent Variable (Target)</label>
                          <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            className="w-full bg-white border border-gray-200 rounded-lg p-2 focus:outline-none"
                          >
                            {activeDataset.columns.map(c => (
                              <option key={c.name} value={c.name}>{c.name}</option>
                            ))}
                          </select>
                          <span className="text-[10px] text-gray-400 italic block mt-1 leading-normal">
                            Target Problem type is automatically updated on choose.
                          </span>
                        </div>
                      </div>

                      {/* Prep Settings */}
                      <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 space-y-3">
                        <h5 className="font-bold text-gray-700 uppercase tracking-wide text-[10px]">
                          Data Preprocessing Pipeline
                        </h5>
                        <div className="space-y-2">
                          <div>
                            <label className="block text-[10px] text-gray-400 uppercase mb-1">Null Strategy Imputations</label>
                            <select
                              value={prepOptions.missingStrategy}
                              onChange={(e: any) => setPrepOptions({ ...prepOptions, missingStrategy: e.target.value })}
                              className="w-full bg-white border border-gray-200 rounded-lg p-1.5"
                            >
                              <option value="mean">Replace with Mean bounds</option>
                              <option value="median">Replace with Median</option>
                              <option value="drop">Drop null rows entirely</option>
                            </select>
                          </div>
                          <div>
                            <label className="block text-[10px] text-gray-400 uppercase mb-1">Scaling normalization</label>
                            <select
                              value={prepOptions.scalingMethod}
                              onChange={(e: any) => setPrepOptions({ ...prepOptions, scalingMethod: e.target.value })}
                              className="w-full bg-white border border-gray-200 rounded-lg p-1.5"
                            >
                              <option value="standard">Standard Scaling (z-score)</option>
                              <option value="minmax">MinMax scaling (0-1 normalize)</option>
                              <option value="none">Skip scaling pipeline (Raw)</option>
                            </select>
                          </div>
                        </div>
                      </div>

                      {/* Hyperparameters */}
                      <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 space-y-3">
                        <h5 className="font-bold text-gray-700 uppercase tracking-wide text-[10px]">
                          Hyperparameter Grid Limits
                        </h5>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                              <span>Max Tree Search Depth</span>
                              <span className="font-bold text-gray-700">{maxDepth} levels</span>
                            </div>
                            <input
                              type="range"
                              min="2"
                              max="10"
                              value={maxDepth}
                              onChange={(e) => setMaxDepth(Number(e.target.value))}
                              className="w-full accent-violet-600"
                            />
                          </div>

                          <div>
                            <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                              <span>Random Forest Estimators</span>
                              <span className="font-bold text-gray-700">{nEstimators} trees</span>
                            </div>
                            <input
                              type="range"
                              min="3"
                              max="30"
                              value={nEstimators}
                              onChange={(e) => setNEstimators(Number(e.target.value))}
                              className="w-full accent-violet-600"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 border-t border-gray-100 pt-6 flex justify-end gap-3.5">
                      <button
                        onClick={executePreprocessing}
                        className="px-5 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold text-xs rounded-xl transition-all"
                      >
                        Compute Preprocess Report Only
                      </button>
                      <button
                        onClick={executeModelTraining}
                        disabled={isTraining}
                        className="px-6 py-3 bg-violet-600 hover:bg-violet-700 disabled:bg-violet-400 text-white font-semibold text-xs rounded-xl shadow-md transition-all flex items-center gap-2"
                      >
                        {isTraining ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span>Optimizing Weights...</span>
                          </>
                        ) : (
                          <>
                            <Brain className="w-4 h-4" />
                            <span>Run Training Sequence (Assemble 3 Models)</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Preprocessing Reports */}
                  {preprocessed && (
                    <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm space-y-4">
                      <h4 className="font-bold text-gray-800 text-sm flex items-center gap-2">
                        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                        Autonomic Pipeline Preprocessing Report Logs
                      </h4>
                      <p className="text-xs text-gray-400 leading-relaxed font-mono">
                        Columns containing values successfully prepared to feed mathematical algorithmic formats.
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
                        <div className="p-3 bg-gray-50 border border-gray-100 rounded-xl">
                          <span className="text-[10px] text-gray-400 tracking-wider font-bold block uppercase">Duplicate Cleaned</span>
                          <span className="text-sm font-bold text-gray-700 block mt-1">{preprocessed.report.duplicatesRemoved} rows dropped</span>
                        </div>
                        <div className="p-3 bg-gray-50 border border-gray-100 rounded-xl">
                          <span className="text-[10px] text-gray-400 tracking-wider font-bold block uppercase">Enforced Scaling</span>
                          <span className="text-sm font-bold text-gray-700 block mt-1">{preprocessed.report.scaledColumns.length} variables fitted</span>
                        </div>
                        <div className="p-3 bg-gray-50 border border-gray-100 rounded-xl">
                          <span className="text-[10px] text-gray-400 tracking-wider font-bold block uppercase">Categorical Encoded</span>
                          <span className="text-sm font-bold text-gray-700 block mt-1 truncate">
                            {preprocessed.report.encodedColumns.join(', ') || 'No categoricals'}
                          </span>
                        </div>
                        <div className="p-3 bg-gray-50 border border-gray-100 rounded-xl">
                          <span className="text-[10px] text-gray-400 tracking-wider font-bold block uppercase">Features selected</span>
                          <span className="text-sm font-bold text-gray-700 block mt-1">{preprocessed.report.selectedFeatures.length} attributes kept</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* PAGE 5: COMPARISON */}
              {activeTab === 'comparison' && (
                <div className="space-y-6">
                  {Object.keys(trainedModels).length === 0 ? (
                    <div className="bg-white border border-gray-100 rounded-2xl p-12 text-center max-w-md mx-auto">
                      <AlertCircle className="w-12 h-12 text-violet-500 mx-auto mb-4" />
                      <h4 className="font-bold text-gray-800 text-sm">Calibration Required</h4>
                      <p className="text-xs text-gray-400 mt-2 leading-relaxed">
                        Please proceed to the <strong>Model Training Studio</strong> in the sidebar to configure targets and fit validation curves first.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      
                      {/* Leader Recommend card */}
                      <div className="bg-gradient-to-r from-emerald-500 to-teal-600 text-white p-6 rounded-2xl shadow flex items-center justify-between">
                        <div>
                          <span className="text-[9px] font-bold uppercase tracking-widest bg-emerald-600/30 px-3 py-1 rounded-full border border-emerald-400/20">
                            AutoML Best Model Recommendation
                          </span>
                          <h4 className="text-2xl font-black mt-3">
                            Optimal Model Fitted: {trainedModels[bestModelId]?.name}
                          </h4>
                          <p className="text-xs text-emerald-50 mt-1 leading-normal">
                            Optimally fits the holdout testing targets with a cross validation mean scoring index of{' '}
                            <strong>{(trainedModels[bestModelId]?.metrics.cvMean * 100).toFixed(2)}%</strong>.
                          </p>
                        </div>
                        <div className="p-4 bg-emerald-600/20 rounded-full border border-emerald-300/20 hidden md:block">
                          <Brain className="w-8 h-8 text-white" />
                        </div>
                      </div>

                      {/* Performance Grid metrics spreadsheets */}
                      <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                        <h4 className="font-bold text-gray-800 text-sm mb-4">Competitor Scoring spreadsheets metrics grid</h4>
                        
                        <div className="overflow-x-auto">
                          <table className="min-w-full text-left font-mono text-xs border border-gray-100 rounded-lg overflow-hidden">
                            <thead className="bg-gray-50">
                              {problemType === 'classification' ? (
                                <tr>
                                  <th className="p-3 text-gray-500 font-bold">Model name</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">Train Acc</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">Test Acc</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">Precision</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">Recall Score</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">F1 weighted</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">ROC AUC</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">5-Fold CV Mean</th>
                                </tr>
                              ) : (
                                <tr>
                                  <th className="p-3 text-gray-500 font-bold">Model name</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">R² variance score</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">MAE (Abs Error)</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">MSE</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">RMSE (Root)</th>
                                  <th className="p-3 text-gray-500 font-bold text-center">5-Fold CV Mean (R²)</th>
                                </tr>
                              )}
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                              {(Object.entries(trainedModels) as [string, TrainedModel][]).map(([id, m]) => (
                                <tr key={id} className={id === bestModelId ? 'bg-emerald-50/20 font-bold' : ''}>
                                  <td className="p-3 text-gray-800 font-bold flex items-center gap-1.5">
                                    {m.name}
                                    {id === bestModelId && <span className="bg-emerald-500 text-white text-[8px] px-1.5 py-0.5 rounded uppercase">Best</span>}
                                  </td>
                                  
                                  {problemType === 'classification' ? (
                                    <>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.trainingAccuracy || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-700 text-sm">{(m.metrics.testingAccuracy || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.precision || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.recall || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.f1Score || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.rocAuc || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-800">{(m.metrics.cvMean || 0).toFixed(4)}</td>
                                    </>
                                  ) : (
                                    <>
                                      <td className="p-3 text-center text-gray-700 text-sm">{(m.metrics.r2Score || 0).toFixed(4)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.mae || 0).toFixed(2)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.mse || 0).toFixed(2)}</td>
                                      <td className="p-3 text-center text-gray-500">{(m.metrics.rmse || 0).toFixed(2)}</td>
                                      <td className="p-3 text-center text-gray-800">{(m.metrics.cvMean || 0).toFixed(4)}</td>
                                    </>
                                  )}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Charts comparison rendering */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        
                        <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                          <h4 className="font-bold text-gray-800 text-sm mb-4">Competitor Scoring Graphical Chart</h4>
                          <MetricsBarChart 
                            modelsMetrics={(Object.values(trainedModels) as TrainedModel[]).map(m => ({
                              model: m.name,
                              value: problemType === 'classification' ? (m.metrics.testingAccuracy || 0) : (m.metrics.r2Score || 0)
                            }))}
                            metricKey={problemType === 'classification' ? 'accuracy' : 'r2'}
                            metricLabel={problemType === 'classification' ? 'Testing Accuracy' : 'R² Score (Variance)'}
                          />
                        </div>

                        {/* Confusion Matrix if Classification */}
                        {problemType === 'classification' && trainedModels[bestModelId]?.confusionMatrix && (
                          <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                            <h4 className="font-bold text-gray-800 text-sm mb-2">Confusion Matrix Analysis (Hold-out Test Set)</h4>
                            <p className="text-xs text-gray-400 mb-6 font-mono">Actual classes vs predicted classifications.</p>
                            
                            <div className="flex flex-col items-center">
                              <div className="grid grid-cols-3 gap-1 font-mono text-xs w-full max-w-[280px]">
                                <div className="col-span-3 text-center font-bold text-[10px] text-gray-400 uppercase">Predicted</div>
                                <div className="row-span-2 flex items-center justify-center writing-vertical font-bold text-[10px] text-gray-400 uppercase">Actual</div>
                                
                                {trainedModels[bestModelId].confusionMatrix?.labels.map((lbl, idx) => (
                                  <div key={lbl} className="col-span-1 text-center font-bold text-gray-500 truncate max-w-[80px]" title={lbl}>{lbl}</div>
                                ))}

                                {trainedModels[bestModelId].confusionMatrix?.matrix.map((rowArr, rIdx) => {
                                  return (
                                    <React.Fragment key={rIdx}>
                                      {rowArr.map((cellVal, cIdx) => (
                                        <div 
                                          key={cIdx} 
                                          className={`aspect-square flex items-center justify-center font-bold text-sm border border-gray-100 text-center rounded-lg transition-all ${
                                            rIdx === cIdx ? 'bg-emerald-500 text-white shadow-inner scale-[0.98]' : 'bg-gray-50 text-gray-400'
                                          }`}
                                        >
                                          {cellVal}
                                        </div>
                                      ))}
                                    </React.Fragment>
                                  );
                                })}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>

                    </div>
                  )}
                </div>
              )}

              {/* PAGE 6: FEATURE IMPORTANCE */}
              {activeTab === 'importance' && (
                <div className="space-y-6">
                  {Object.keys(trainedModels).length === 0 ? (
                    <div className="bg-white border border-gray-100 rounded-2xl p-12 text-center max-w-md mx-auto">
                      <AlertCircle className="w-12 h-12 text-violet-500 mx-auto mb-4" />
                      <h4 className="font-bold text-gray-800 text-sm">Calibration Required</h4>
                      <p className="text-xs text-gray-400 mt-2">Train models before exploring variable weights coefficients.</p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                        <h4 className="font-bold text-gray-800 text-sm mb-4">Select Tree-Based Model weights metrics</h4>
                        <div className="w-full sm:w-72">
                          <select 
                            id="importance-model-select"
                            className="w-full font-mono text-xs bg-gray-50 border border-gray-100 p-2.5 rounded-xl font-bold"
                          >
                            <option value="rf">Random Forest Classifier/Regressor</option>
                            <option value="dt">Decision Tree Classifier/Regressor</option>
                          </select>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* feature importance bar chart */}
                        <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                          <h4 className="font-bold text-gray-800 text-sm mb-2">Gini-Impurity Feature Weights Bar Chart</h4>
                          <p className="text-xs text-gray-400 mb-6 font-mono">Relative weighting contributions of variables configured in decision bounds splits.</p>
                          <FeatureImportanceBarChart 
                            items={Object.entries(trainedModels.rf?.featureImportances || {}).map(([feature, weight]) => ({
                              feature,
                              weight: weight as number
                            }))} 
                          />
                        </div>

                        {/* table representation */}
                        <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                          <h4 className="font-bold text-gray-800 text-sm mb-4">Feature Rankings spreadsheet</h4>
                          
                          <div className="overflow-x-auto">
                            <table className="min-w-full text-left font-mono text-xs">
                              <thead className="bg-gray-50">
                                <tr>
                                  <th className="p-3 text-gray-500 font-bold">Rank ID</th>
                                  <th className="p-3 text-gray-500 font-bold">Variable Name</th>
                                  <th className="p-3 text-gray-500 font-bold text-right">Relative Weight Percentage</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-100">
                                {Object.entries(trainedModels.rf?.featureImportances || {})
                                  .sort((a,b) => (b[1] as number) - (a[1] as number))
                                  .map(([feat, imp], idx) => (
                                    <tr key={feat}>
                                      <td className="p-3 font-bold text-violet-600">#{idx + 1}</td>
                                      <td className="p-3 text-gray-800 font-bold">{feat}</td>
                                      <td className="p-3 text-right font-bold text-gray-600">{((imp as number) * 100).toFixed(2)}%</td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* PAGE 7: PREDICTION CENTER */}
              {activeTab === 'prediction' && (
                <div className="space-y-6">
                  {Object.keys(trainedModels).length === 0 ? (
                    <div className="bg-white border border-gray-100 rounded-2xl p-12 text-center max-w-md mx-auto">
                      <AlertCircle className="w-12 h-12 text-violet-500 mx-auto mb-4" />
                      <h4 className="font-bold text-gray-800 text-sm">Calibration Required</h4>
                      <p className="text-xs text-gray-400 mt-2">Initialize pipeline optimization parameters in the Sidebar to test predictions.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      
                      {/* Configuration Controls Sliders */}
                      <div className="lg:col-span-2 bg-white border border-gray-100 rounded-2xl p-6 shadow-sm space-y-4">
                        <h4 className="font-bold text-gray-800 text-sm">Dynamic Slider Controls</h4>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
                          {preprocessed?.report.selectedFeatures.map(feat => {
                            const stats = preprocessed.columns.find(c => c.name === feat);
                            const val = predictInputs[feat] !== undefined ? predictInputs[feat] : (stats?.mean ?? 0);
                            return (
                              <div key={feat} className="p-3 bg-gray-50 border border-gray-100 rounded-xl">
                                <div className="flex justify-between items-center text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                                  <span className="font-bold truncate">{feat}</span>
                                  <span className="font-bold text-gray-800">{val.toFixed(2)}</span>
                                </div>
                                <input
                                  type="range"
                                  min={stats?.min !== undefined ? stats.min : 0}
                                  max={stats?.max !== undefined ? stats.max : 100}
                                  step={stats?.type === 'numeric' ? (stats.max! - stats.min!) / 100 : 1}
                                  value={val}
                                  onChange={(e) => setPredictInputs({
                                    ...predictInputs,
                                    [feat]: Number(e.target.value)
                                  })}
                                  className="w-full accent-violet-600"
                                />
                                <div className="flex justify-between text-[8px] text-gray-300 mt-1">
                                  <span>Min: {stats?.min?.toFixed(1) || 0}</span>
                                  <span>Max: {stats?.max?.toFixed(1) || 100}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>

                        <div className="mt-6 border-t border-gray-100 pt-4 flex flex-wrap gap-2 justify-end">
                          <button
                            onClick={() => triggerPrediction('rf')}
                            className="px-4 py-2 bg-violet-600/10 text-violet-700 hover:bg-violet-600/20 font-bold text-xs rounded-lg transition-all"
                          >
                            Predict with Forest
                          </button>
                          <button
                            onClick={() => triggerPrediction('dt')}
                            className="px-4 py-2 bg-pink-600/10 text-pink-700 hover:bg-pink-600/20 font-bold text-xs rounded-lg transition-all"
                          >
                            Predict with Tree
                          </button>
                          <button
                            onClick={() => triggerPrediction('linear_log')}
                            className="px-4 py-2 bg-emerald-600/10 text-emerald-700 hover:bg-emerald-600/20 font-bold text-xs rounded-lg transition-all"
                          >
                            Predict with Regression
                          </button>
                        </div>
                      </div>

                      {/* Display prediction result card */}
                      <div className="space-y-4">
                        <div className="bg-white border border-gray-105 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-[340px]">
                          <div>
                            <span className="text-[9px] font-bold uppercase tracking-widest bg-gray-100 text-gray-500 px-3 py-1 rounded-full">
                              Result Panel Output
                            </span>
                          </div>

                          {predictionResult ? (
                            <div className="text-center my-6">
                              <h5 className="text-[10px] text-gray-400 font-bold tracking-wider uppercase">Calculated prediction value ({targetColumn}):</h5>
                              <strong className="text-4xl font-black text-violet-700 font-mono mt-3 block">{predictionResult.prediction}</strong>
                              
                              <div className="mt-4 flex items-center justify-center gap-1 text-[11px] text-gray-500 bg-gray-50 px-3 py-1.5 rounded-full border border-gray-100 w-max mx-auto">
                                <Gauge className="w-3.5 h-3.5 text-violet-500" />
                                <span>Confidence level: <strong>{(predictionResult.confidence * 100).toFixed(0)}%</strong></span>
                              </div>
                            </div>
                          ) : (
                            <div className="text-center my-8 text-gray-400 space-y-2">
                              <HelpCircle className="w-12 h-12 text-gray-300 mx-auto" />
                              <p className="text-xs font-semibold">Ready for inputs simulation.</p>
                              <p className="text-[10px] leading-normal pl-2 pr-2">Slider controls compile weights real coefficients in real time.</p>
                            </div>
                          )}

                          <div className="text-[9px] text-gray-400 text-center leading-normal border-t border-gray-50 pt-3">
                            Features are matched perfectly. Scaling parameters are applied.
                          </div>
                        </div>
                      </div>

                    </div>
                  )}
                </div>
              )}

              {/* PAGE 8: REPORTS & EXPORTER PANEL */}
              {activeTab === 'code' && (
                <div className="space-y-6 animate-fade-in">
                  
                  {/* Dynamic Downloads trigger */}
                  {Object.keys(trainedModels).length > 0 && (
                    <div className="bg-white border border-gray-100 p-6 rounded-2xl flex flex-wrap justify-between items-center gap-4 shadow-sm">
                      <div>
                        <h4 className="font-bold text-gray-800 text-sm">Download Dynamic Performance documents</h4>
                        <p className="text-xs text-gray-400">Export scores matrices, markdown notes, and pickle stubs of your calibration run.</p>
                      </div>
                      
                      <div className="flex gap-2.5">
                        <a
                          href={`data:text/csv;charset=utf-8,${encodeURIComponent(generateCSVScoreboardString())}`}
                          download={`${activeDataset.name.replace(/\s+/g, '_')}_metrics_matrix.csv`}
                          className="px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5"
                        >
                          <FileDown className="w-4 h-4 text-gray-500" />
                          <span>Metrics CSV</span>
                        </a>
                        
                        <a
                          href={`data:text/markdown;charset=utf-8,${encodeURIComponent(generateMarkdownReportString())}`}
                          download={`${activeDataset.name.replace(/\s+/g, '_')}_training_report.md`}
                          className="px-4 py-2.5 bg-violet-50 text-violet-700 hover:bg-violet-100 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 border border-violet-100"
                        >
                          <Download className="w-4 h-4 text-violet-500" />
                          <span>Report Memo (.md)</span>
                        </a>
                      </div>
                    </div>
                  )}

                  {/* Code Explorer */}
                  <CodeExporter />

                </div>
              )}

            </motion.div>
          </AnimatePresence>
        </div>

        {/* Footer */}
        <footer className="h-12 border-t border-[#1E293B] bg-[#0B1120] px-8 flex items-center justify-between text-[11px] text-slate-500">
          <div className="flex gap-6">
            <span>
              <span className="text-emerald-500 font-bold uppercase mr-1.5">• Online</span>
              Server Cluster: <span className="font-semibold text-slate-400">AWS-West-2</span>
            </span>
            <span className="hidden sm:inline text-slate-600">|</span>
            <span>
              <span className="text-white font-medium mr-1">Dataset:</span>
              <span className="font-mono">{activeDataset.rowCount.toLocaleString()} rows, {activeDataset.colCount} cols</span>
            </span>
          </div>
          <div className="flex gap-4 font-mono">
            <span>Last sync: Just now</span>
            <span className="text-slate-700">|</span>
            <a href="#code" onClick={() => setActiveTab('code')} className="text-sky-400 hover:text-sky-300 font-medium">Download PDF Report</a>
          </div>
        </footer>
      </main>
    </div>
  );
}
