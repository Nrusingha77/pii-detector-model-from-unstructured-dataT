
import React from 'react';
import { Shield, FileText, Lock } from 'lucide-react';

const InfoSection = () => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 h-full flex flex-col">
      <h2 className="text-xl md:text-2xl font-semibold text-pii-dark mb-4">How PII Shield Works</h2>
      
      <div className="space-y-4 flex-grow">
        <div className="flex items-start space-x-3">
          <div className="bg-pii-light p-2 rounded-full">
            <FileText className="h-5 w-5 text-pii-primary" />
          </div>
          <div>
            <h3 className="font-medium text-gray-900">Upload Your Document</h3>
            <p className="text-gray-600 text-sm">Upload any document or image containing sensitive information.</p>
          </div>
        </div>
        
        <div className="flex items-start space-x-3">
          <div className="bg-pii-light p-2 rounded-full">
            <Shield className="h-5 w-5 text-pii-primary" />
          </div>
          <div>
            <h3 className="font-medium text-gray-900">Automatic PII Detection</h3>
            <p className="text-gray-600 text-sm">Our system automatically identifies personal information like names, addresses, phone numbers, and more.</p>
          </div>
        </div>
        
        <div className="flex items-start space-x-3">
          <div className="bg-pii-light p-2 rounded-full">
            <Lock className="h-5 w-5 text-pii-primary" />
          </div>
          <div>
            <h3 className="font-medium text-gray-900">Secure Masking</h3>
            <p className="text-gray-600 text-sm">We mask or remove the identified private information while preserving the document's structure and readability.</p>
          </div>
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-100">
        <p className="text-sm text-gray-500">
          All processing happens in your browser. Your files are never stored on our servers, ensuring maximum privacy and security.
        </p>
      </div>
    </div>
  );
};

export default InfoSection;
