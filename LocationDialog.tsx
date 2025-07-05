
import React from 'react';
import { MapPin, Mail, Clock } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface LocationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const LocationDialog = ({ open, onOpenChange }: LocationDialogProps) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-3xl font-bold text-pii-dark">Our Location</DialogTitle>
          <DialogDescription className="text-gray-600">
            Visit our office or get in touch with us
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-6 space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-pii-dark flex items-center">
              <MapPin className="h-5 w-5 mr-2 text-pii-primary" />
              Office Address
            </h2>
            
            <div className="space-y-4 text-gray-700">
              <p>
                GIET University<br />
                Gunupur 765022<br />
                INDIA
              </p>
              
              <div>
                <h3 className="font-medium flex items-center mb-2">
                  <Mail className="h-4 w-4 mr-2 text-pii-primary" />
                  Email
                </h3>
                <p>22cseaiml009.amanraj@giet.edu</p>
              </div>
              
              <div>
                <h3 className="font-medium flex items-center mb-2">
                  <Clock className="h-4 w-4 mr-2 text-pii-primary" />
                  Business Hours
                </h3>
                <p>Monday-Friday: 9:00 AM - 5:00 PM</p>
                <p>Saturday-Sunday: Closed</p>
              </div>
            </div>
          </div>
          
          <div className="w-full h-80 rounded-lg overflow-hidden shadow-md">
            <iframe 
              src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3797.4169203508407!2d83.8279463!3d19.0482583!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2zMTnCsDAyJzUzLjciTiA4M8KwNDknNTQuNiJF!5e0!3m2!1sen!2sin!4v1712657598183!5m2!1sen!2sin" 
              width="100%" 
              height="100%" 
              style={{ border: 0 }} 
              allowFullScreen 
              loading="lazy" 
              referrerPolicy="no-referrer-when-downgrade"
              title="GIET University Map"
              className="w-full h-full"
            ></iframe>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default LocationDialog;
