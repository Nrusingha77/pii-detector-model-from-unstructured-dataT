
import React from 'react';
import { Shield, FileCheck, Lock, Users, BarChart, Zap } from 'lucide-react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

const ServiceCard = ({ icon: Icon, title, description }: { 
  icon: React.ElementType, 
  title: string, 
  description: string 
}) => (
  <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow">
    <div className="bg-pii-light p-3 rounded-full inline-flex mb-4">
      <Icon className="h-6 w-6 text-pii-primary" />
    </div>
    <h3 className="text-lg font-semibold mb-2 text-gray-900">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </div>
);

const Services = () => {
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
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-4xl font-bold text-pii-dark mb-4">Our Services</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            We offer a comprehensive suite of tools to help you identify, mask, and manage personally identifiable information in your documents.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service, index) => (
            <ServiceCard 
              key={index} 
              icon={service.icon} 
              title={service.title} 
              description={service.description} 
            />
          ))}
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Services;
