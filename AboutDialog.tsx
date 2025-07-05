import React from 'react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
interface AboutDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}
const AboutDialog = ({
  open,
  onOpenChange
}: AboutDialogProps) => {
  return <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-3xl font-bold text-pii-dark">About Us</DialogTitle>
          <DialogDescription className="text-gray-600">
            Learn more about PII Shield and our mission
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-6">
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
                  <AvatarImage alt="Aman Raj" className="object-cover" src="/lovable-uploads/dd876f65-24c9-406e-b72f-42417ad462a4.png" />
                  <AvatarFallback>AR</AvatarFallback>
                </Avatar>
                <h3 className="text-xl font-semibold text-pii-primary">AMAN RAJ</h3>
              </div>
              <div className="flex flex-col items-center">
                <Avatar className="h-48 w-48 mb-4">
                  <AvatarImage alt="Nrusingha Prasad Khadanga" className="object-cover" src="/lovable-uploads/dd623973-7c0c-4125-8bda-ab626d36dc2c.png" />
                  <AvatarFallback>NK</AvatarFallback>
                </Avatar>
                <h3 className="text-xl font-semibold text-pii-primary text-center">NRUSINGHA PRASAD<br />KHADANGA</h3>
              </div>
              <div className="flex flex-col items-center">
                <Avatar className="h-48 w-48 mb-4">
                  <AvatarImage alt="Mukunda Raja Saha" src="/lovable-uploads/2d9e18cc-d3fe-499f-a275-094ef429dcff.png" className="object-fill" />
                  <AvatarFallback>MS</AvatarFallback>
                </Avatar>
                <h3 className="text-xl font-semibold text-pii-primary">MUKUNDA RAJA SAHA</h3>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>;
};
export default AboutDialog;