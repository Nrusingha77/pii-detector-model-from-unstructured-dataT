
import React from 'react';
import { Shield, FileCheck, Lock, Users, BarChart, Zap } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface ServiceCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
}

const ServiceCard = ({ icon: Icon, title, description }: ServiceCardProps) => (
  <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow">
    <div className="bg-pii-light p-3 rounded-full inline-flex mb-4">
      <Icon className="h-6 w-6 text-pii-primary" />
    </div>
    <h3 className="text-lg font-semibold mb-2 text-gray-900">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

interface ServiceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ServiceDialog = ({ open, onOpenChange }: ServiceDialogProps) => {
  const services = [
    {
      icon: Shield,
      title: "PII Detection & Masking",
      description: "Automatically identify and mask personally identifiable information in documents and images."
    },
    {
      icon: FileCheck,
      title: "Document Compliance",
      description: "Ensure your documents comply with GDPR, HIPAA, and other privacy regulations."
    },
    {
      icon: Lock,
      title: "Secure Processing",
      description: "All processing happens locally in your browser. Your files never leave your device."
    },
    {
      icon: Users,
      title: "Bulk Processing",
      description: "Process multiple documents at once with our batch processing feature."
    },
    {
      icon: BarChart,
      title: "PII Analytics",
      description: "Gain insights into the types and frequency of PII in your documents."
    },
    {
      icon: Zap,
      title: "API Integration",
      description: "Integrate our PII detection and masking capabilities into your own applications."
    }
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-3xl font-bold text-pii-dark">Our Services</DialogTitle>
          <DialogDescription className="text-gray-600">
            We offer a comprehensive suite of tools to help you identify, mask, and manage personally identifiable information in your documents.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          {services.map((service, index) => (
            <ServiceCard 
              key={index} 
              icon={service.icon} 
              title={service.title} 
              description={service.description} 
            />
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ServiceDialog;
