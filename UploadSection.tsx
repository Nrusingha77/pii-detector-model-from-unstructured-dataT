import React, { useState, useRef } from 'react';
import { Upload, File, X, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import axios from 'axios';

const EXPRESS_API_URL = 'http://localhost:5000'; // Add to environment variables

const UploadSection = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedFileUrl, setProcessedFileUrl] = useState<string | null>(null);
  const [processedFileName, setProcessedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    // Update allowed types to match backend
    const allowedTypes = [
      'text/plain',
      'text/csv',
      'message/rfc822',
      'application/pdf'
    ];

    if (!allowedTypes.includes(file.type)) {
      toast({
        title: "Unsupported file type",
        description: "Please upload a TXT, CSV, or EMAIL file.",
        variant: "destructive"
      });
      return;
    }

    setFile(file);
    setProcessedFileUrl(null);
  };

  const removeFile = () => {
    setFile(null);
    setProcessedFileUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const processFile = async () => {
    if (!file) return;
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('mode', 'mask'); // or 'remove' based on user choice

      // Send to Express API
      interface ProcessResponse {
        success: boolean;
        pdfUrl: string;
        data: { message: string };
      }

      const response = await axios.post<ProcessResponse>(
        `${EXPRESS_API_URL}/api/pii/process`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          }
        }
      );

      if (response.data.success) {
        setProcessedFileUrl(response.data.pdfUrl);
        setProcessedFileName(response.data.data.message);

        toast({
          title: "Processing complete",
          description: "Your document has been processed successfully."
        });
      }
    } catch (error) {
      console.error('Error processing file:', error);
      toast({
        title: "Processing failed",
        description: "There was an error processing your document.",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadFile = async () => {
    if (!processedFileUrl) return;

    try {
      // Add error handling for response status
      const response = await axios.get(processedFileUrl, {
        responseType: 'blob',
        validateStatus: (status) => status === 200
      });

      // Create blob URL and trigger download
      const blob = new Blob([response.data as BlobPart], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `processed_${file?.name}.pdf`);
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }, 100);

      toast({
        title: "Download started",
        description: "Your processed document is being downloaded."
      });
    } catch (error) {
      console.error('Error downloading file:', error);
      toast({
        title: "Download failed",
        description: "Failed to download the processed document.",
        variant: "destructive"
      });
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 h-full flex flex-col">
      <h2 className="text-xl md:text-2xl font-semibold text-pii-dark mb-4">Upload Your Document</h2>

      <div className={`flex-grow flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-6 
          ${isDragging ? 'border-pii-accent bg-pii-light' : 'border-gray-300'} 
          ${file ? 'bg-pii-light/30' : 'bg-gray-50'} 
          transition-colors cursor-pointer`} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}>
        <input type="file" ref={fileInputRef} onChange={handleFileInput} className="hidden" accept=".txt,.csv,.eml,.pdf" />

        {!file ? (
          <>
            <Upload className={`h-12 w-12 text-gray-400 mb-4 ${isDragging ? 'animate-bounce-light' : ''}`} />
            <p className="text-center text-gray-600 mb-2">
              <span className="font-medium">CLICK HERE TO UPLOAD A DOCUMENT</span>
            </p>
            <p className="text-center text-gray-500 text-sm">
              or drag and drop<br />
              (TXT, CSV, EMAIL, PDF)
            </p>
          </>
        ) : (
          <div className="w-full">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <File className="h-8 w-8 text-pii-primary mr-3" />
                <div>
                  <p className="font-medium text-gray-900 truncate max-w-[200px]">{file.name}</p>
                  <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <button onClick={e => {
                e.stopPropagation();
                removeFile();
              }} className="p-1 rounded-full hover:bg-gray-200">
                <X className="h-5 w-5 text-gray-500" />
              </button>
            </div>

            <Button className="w-full bg-pii-primary hover:bg-pii-secondary" onClick={e => {
              e.stopPropagation();
              processFile();
            }} disabled={isProcessing}>
              {isProcessing ? "Processing..." : "Process Document"}
            </Button>
          </div>
        )}
      </div>

      {processedFileUrl && (
        <div className="mt-4 bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-medium text-gray-900">Processed Document</h3>
            <Button 
              onClick={downloadFile}
              className="flex items-center gap-2 bg-pii-primary hover:bg-pii-secondary"
            >
              <Download size={18} />
              Download PDF
            </Button>
          </div>
          <div className="p-3 bg-white rounded border border-gray-300">
            <p className="text-gray-700">
              Your document has been processed and converted to PDF.
              All sensitive information has been protected.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadSection;
