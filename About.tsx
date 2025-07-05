
import React from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const About = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-pii-dark mb-4">About Us</h1>
            <div className="h-1 w-20 bg-pii-primary mx-auto"></div>
          </div>
          
          <div className="bg-white p-8 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4 text-pii-dark">Our Mission</h2>
            <p className="text-gray-700 mb-6">
              At PII Shield, our mission is to protect privacy by making it easy for anyone to identify and mask personally identifiable information. 
              In a world where data privacy concerns are growing, we believe that everyone should have access to tools that help them protect sensitive information.
            </p>
            
            <h2 className="text-2xl font-semibold mb-4 text-pii-dark">Our Approach</h2>
            <p className="text-gray-700 mb-6">
              We've built PII Shield with privacy as our core principle. That's why all document processing happens locally in your browser – your files never 
              leave your device. This client-side approach ensures maximum security while delivering powerful PII detection and masking capabilities.
            </p>
            
            <div className="bg-pii-light p-6 rounded-lg mt-8">
              <h3 className="text-lg font-semibold mb-2 text-pii-dark">Our Privacy Commitment</h3>
              <ul className="list-disc pl-5 text-gray-700 space-y-2">
                <li>We never store your documents on our servers</li>
                <li>All processing happens locally in your browser</li>
                <li>We don't track what information is in your documents</li>
                <li>We follow best practices for web security</li>
              </ul>
            </div>

            <div className="mt-12">
              <h2 className="text-3xl font-bold text-center text-pii-primary mb-8">Meet Our Team</h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="flex flex-col items-center">
                  <Avatar className="h-48 w-48 mb-4">
                    <AvatarImage src="/lovable-uploads/1a551bdf-b745-49a3-a53e-9d4aafb16a9d.png" alt="Aman Raj" className="object-cover" />
                    <AvatarFallback>AR</AvatarFallback>
                  </Avatar>
                  <h3 className="text-xl font-semibold text-pii-primary">AMAN RAJ</h3>
                </div>
                <div className="flex flex-col items-center">
                  <Avatar className="h-48 w-48 mb-4">
                    <AvatarImage src="/lovable-uploads/1a551bdf-b745-49a3-a53e-9d4aafb16a9d.png" alt="Nrusingha Prasad Khadanga" className="object-cover" />
                    <AvatarFallback>NK</AvatarFallback>
                  </Avatar>
                  <h3 className="text-xl font-semibold text-pii-primary text-center">NRUSINGHA PRASAD<br />KHADANGA</h3>
                </div>
                <div className="flex flex-col items-center">
                  <Avatar className="h-48 w-48 mb-4">
                    <AvatarImage src="/lovable-uploads/1a551bdf-b745-49a3-a53e-9d4aafb16a9d.png" alt="Mukunda Raja Saha" className="object-cover" />
                    <AvatarFallback>MS</AvatarFallback>
                  </Avatar>
                  <h3 className="text-xl font-semibold text-pii-primary">MUKUNDA RAJA SAHA</h3>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default About;
