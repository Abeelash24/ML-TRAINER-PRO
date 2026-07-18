/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { pythonFiles, PythonCodeFile } from '../data/pythonCodeTemplates';
import { 
  FileCode, 
  Terminal, 
  Check, 
  Copy, 
  Download, 
  Folder, 
  Info, 
  BookOpen, 
  FolderHeart,
  ChevronRight,
  Server
} from 'lucide-react';
import { motion } from 'motion/react';

export default function CodeExporter() {
  const [selectedFile, setSelectedFile] = useState<PythonCodeFile>(pythonFiles[0]);
  const [activeTab, setActiveTab] = useState<'code' | 'guide' | 'deploy'>('code');
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadSingle = (file: PythonCodeFile) => {
    const blob = new Blob([file.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadAll = () => {
    // Trigger download for each of the files
    pythonFiles.forEach(file => {
      // Create relative path hierarchy indicators inside download names to help them organize
      const filename = file.path.includes('/') 
        ? `${file.path.split('/')[0]}_${file.name}` 
        : file.name;

      const blob = new Blob([file.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    });
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden min-h-[500px]">
      {/* Top Banner Options */}
      <div className="bg-gray-50 border-b border-gray-100 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div>
          <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
            <Terminal className="text-violet-500 w-5 h-5" />
            Python & Streamlit Code Export Center
          </h2>
          <p className="text-xs text-gray-400">Download files representing the complete local execution logic.</p>
        </div>
        <div className="flex bg-gray-200/60 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('code')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'code' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" /> Source Files
          </button>
          <button
            onClick={() => setActiveTab('guide')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'guide' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" /> Setup Guide
          </button>
          <button
            onClick={() => setActiveTab('deploy')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'deploy' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            <Server className="w-3.5 h-3.5" /> Live Deploy
          </button>
        </div>
      </div>

      {activeTab === 'code' && (
        <div className="flex flex-col lg:flex-row min-h-[560px]">
          {/* Left panel folders */}
          <div className="w-full lg:w-72 bg-gray-50/50 border-r border-gray-100 p-4 flex flex-col justify-between">
            <div>
              <span className="text-xs font-bold text-gray-400 uppercase tracking-widest pl-2">Project Workspace</span>
              <div className="mt-3 space-y-1">
                {/* Project workspace folder visual */}
                <div className="flex items-center gap-2 px-3 py-2 text-xs font-bold text-gray-700 bg-gray-100 rounded-lg">
                  <Folder className="w-4 h-4 text-violet-500 fill-violet-200" />
                  <span>machine_learning_studio/</span>
                </div>

                <div className="pl-4 space-y-1 mt-1 border-l border-gray-200 ml-5">
                  {pythonFiles.map(file => {
                    const isSelected = file.path === selectedFile.path;
                    const pathParts = file.path.split('/');
                    return (
                      <button
                        key={file.path}
                        onClick={() => setSelectedFile(file)}
                        className={`w-full text-left px-3 py-2 text-xs font-medium rounded-lg flex items-center transition-all ${
                          isSelected 
                            ? 'bg-violet-50 text-violet-700 shadow-sm' 
                            : 'text-gray-500 hover:bg-gray-100 hover:text-gray-700'
                        }`}
                      >
                        {pathParts.length > 1 ? (
                          <div className="flex items-center gap-1.5 w-full">
                            <span className="text-[10px] text-gray-300 font-mono">utils/</span>
                            <span className="truncate">{file.name}</span>
                          </div>
                        ) : (
                          <span className="truncate">{file.name}</span>
                        )}
                        <ChevronRight className={`w-3 h-3 ml-auto opacity-70 ${isSelected ? 'block' : 'hidden'}`} />
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="mt-8 border-t border-gray-200 pt-4">
              <button
                onClick={handleDownloadAll}
                className="w-full px-4 py-3 bg-violet-600 hover:bg-violet-700 text-white font-semibold text-xs rounded-xl shadow-md transition-all flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Complete Code
              </button>
              <p className="text-[10px] text-gray-400 text-center mt-2.5 leading-relaxed pl-1">
                Downloads all 6 modular scripts mapped in structural layouts.
              </p>
            </div>
          </div>

          {/* Code Viewer Panel */}
          <div className="flex-1 p-6 bg-gray-900 border-t lg:border-t-0 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center pb-3 border-b border-gray-800">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span className="text-xs text-gray-400 font-mono ml-2 pl-2 border-l border-gray-800">
                    {selectedFile.path}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopy}
                    className="p-1 px-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg transition-all flex items-center gap-1 text-xs"
                    title="Copy to Clipboard"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-emerald-400">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => handleDownloadSingle(selectedFile)}
                    className="p-1 px-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-all flex items-center gap-1 text-xs font-semibold"
                    title="Download individual file"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download</span>
                  </button>
                </div>
              </div>

              {/* Code text frame */}
              <div className="mt-4 overflow-auto max-h-[460px]">
                <pre className="text-xs text-emerald-300 font-mono leading-relaxed whitespace-pre font-medium pl-1">
                  <code>{selectedFile.content}</code>
                </pre>
              </div>
            </div>
            
            <div className="mt-4 pt-4 border-t border-gray-800 flex items-center gap-2 text-gray-500 text-xs">
              <Info className="w-4 h-4 text-violet-400 flex-shrink-0" />
              <span>Use the downloaded structure by placing helper files inside a dedicated <code className="text-violet-300">utils/</code> subdirectory alongside <code className="text-violet-300">app.py</code>.</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'guide' && (
        <div className="p-8 prose max-w-none text-gray-600 text-sm">
          <div className="max-w-3xl mx-auto space-y-6">
            <h3 className="text-xl font-bold text-gray-800 border-b pb-2 flex items-center gap-2">
              <FolderHeart className="w-5 h-5 text-rose-500 fill-rose-100" />
              Step-by-step Project Execution Guide
            </h3>
            
            <p className="leading-relaxed">
              This system is fully ready for college presentations. Follow these instructions to set up the workspace locally and compile:
            </p>

            <div className="space-y-4">
              <div className="flex gap-4">
                <span className="w-6 h-6 flex-shrink-0 flex items-center justify-center rounded-full bg-violet-100 text-violet-700 font-mono font-bold text-xs mt-0.5">
                  1
                </span>
                <div>
                  <h4 className="font-bold text-gray-800">Establish the Project Folders</h4>
                  <p className="text-xs text-gray-500 mt-1">Create a parent folder on your machine and map the downloaded script files cleanly in subdirectories:</p>
                  <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs font-mono text-gray-700 border border-gray-100 leading-normal">
{`project_folder/
├── app.py
├── requirements.txt
└── utils/
    ├── __init__.py      # Empty file to indicate python module
    ├── preprocessing.py
    ├── visualization.py
    ├── model_training.py
    └── report_generator.py`}
                  </pre>
                </div>
              </div>

              <div className="flex gap-4">
                <span className="w-6 h-6 flex-shrink-0 flex items-center justify-center rounded-full bg-violet-100 text-violet-700 font-mono font-bold text-xs mt-0.5">
                  2
                </span>
                <div>
                  <h4 className="font-bold text-gray-800">Configure a Python Virtual Environment</h4>
                  <p className="text-xs text-gray-500 mt-1">Open your local terminal in the parent project directory and setup a fresh isolated environment to avoid library leaks:</p>
                  <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs font-mono text-gray-700 border border-gray-100">
{`# Create virtualenv
python -m venv venv

# Activate Environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\\Scripts\\activate`}
                  </pre>
                </div>
              </div>

              <div className="flex gap-4">
                <span className="w-6 h-6 flex-shrink-0 flex items-center justify-center rounded-full bg-violet-100 text-violet-700 font-mono font-bold text-xs mt-0.5">
                  3
                </span>
                <div>
                  <h4 className="font-bold text-gray-800">Install Dependent Libraries</h4>
                  <p className="text-xs text-gray-500 mt-1">Run pip on your machine to install pandas, plotly, scikit-learn, and streamlit requirements:</p>
                  <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs font-mono text-gray-700 border border-gray-100">
pip install -r requirements.txt
                  </pre>
                </div>
              </div>

              <div className="flex gap-4">
                <span className="w-6 h-6 flex-shrink-0 flex items-center justify-center rounded-full bg-violet-100 text-violet-700 font-mono font-bold text-xs mt-0.5">
                  4
                </span>
                <div>
                  <h4 className="font-bold text-gray-800">Run the ML Training Studio App</h4>
                  <p className="text-xs text-gray-500 mt-1">Boot up the local browser portal with Streamlit directly from the active shell:</p>
                  <pre className="mt-2 bg-gray-50 p-3 rounded-lg text-xs font-mono text-gray-700 border border-gray-100">
streamlit run app.py
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-blue-800 flex gap-3 text-xs">
              <Info className="w-4.5 h-4.5 text-blue-500 flex-shrink-0" />
              <div>
                <span className="font-bold">Demonstration Tip:</span> When showcasing, make sure to show how changes in the Sidebar Preprocessing parameters (like missing imputations or outlier clippings) dynamically update test accuracies and confusion matrices in real-time over the metrics cards!
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'deploy' && (
        <div className="p-8 prose max-w-none text-gray-600 text-sm">
          <div className="max-w-3xl mx-auto space-y-6">
            <h3 className="text-xl font-bold text-gray-800 border-b pb-2 flex items-center gap-2">
              <Server className="w-5 h-5 text-emerald-500" />
              Streamlit Community Cloud Deployment Guide
            </h3>
            
            <p className="leading-relaxed">
              Streamlit offers high-utility free cloud hosting for repository applications. Follow this workflow to deploy your custom model studio live on the web:
            </p>

            <div className="space-y-4">
              <div className="bg-gray-50 p-5 rounded-xl border border-gray-100">
                <h4 className="font-bold text-gray-800 flex items-center gap-1.5 text-xs uppercase text-violet-700 tracking-wider">
                  Phase 1: Sync to GitHub
                </h4>
                <ul className="list-disc pl-5 text-xs text-gray-500 mt-2 space-y-1.5">
                  <li>Initiate a new public repository on GitHub (e.g., name it <code className="bg-gray-200/50 px-1 rounded text-violet-600">ml-training-studio</code>).</li>
                  <li>Run <code className="bg-gray-200/50 px-1 rounded font-mono">git init</code> in your local folder, add all files, and commit them.</li>
                  <li>Push your commits to your GitHub main branch.</li>
                </ul>
              </div>

              <div className="bg-gray-50 p-5 rounded-xl border border-gray-100">
                <h4 className="font-bold text-gray-800 flex items-center gap-1.5 text-xs uppercase text-emerald-700 tracking-wider">
                  Phase 2: Establish Streamlit Connection
                </h4>
                <ul className="list-disc pl-5 text-xs text-gray-500 mt-2 space-y-1.5">
                  <li>Navigate to <a href="https://share.streamlit.io" target="_blank" rel="noopener noreferrer" className="text-violet-600 underline font-bold">share.streamlit.io</a> and create a free developer account using your GitHub login.</li>
                  <li>Click **"New App"** in your Cloud workspace.</li>
                  <li>Select your repository, the main branch, and type <code className="bg-gray-100 px-1 rounded font-mono">app.py</code> as the main file path.</li>
                  <li>Click **"Deploy!"**</li>
                </ul>
              </div>

              <div className="bg-emerald-50 border border-emerald-100 rounded-xl p-4 text-emerald-800 flex gap-3 text-xs">
                <Check className="w-4.5 h-4.5 text-emerald-500 flex-shrink-0" />
                <div>
                  <span className="font-bold">Live Monitoring:</span> Streamlit will automatically read your <code className="font-semibold">requirements.txt</code>, download packages from pip, run the web application, and generate a shareable web link to distribute to examiners!
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
