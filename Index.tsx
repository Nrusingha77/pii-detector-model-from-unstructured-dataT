
import React from 'react';
import Navbar from '@/components/Navbar';
import InfoSection from '@/components/InfoSection';
import UploadSection from '@/components/UploadSection';
import Footer from '@/components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <InfoSection />
          <UploadSection />
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
