
import React from 'react';
import { MapPin, Phone } from 'lucide-react';
import { Button } from '@/components/ui/button';
import LocationDialog from './LocationDialog';
import ContactDialog from './ContactDialog';

const Footer = () => {
  const [locationDialogOpen, setLocationDialogOpen] = React.useState(false);
  const [contactDialogOpen, setContactDialogOpen] = React.useState(false);

  return (
    <footer className="w-full py-4 px-4 md:px-8 bg-white border-t border-gray-200 mt-auto">
      <div className="container mx-auto flex flex-col sm:flex-row justify-between items-center">
        <div className="flex space-x-4 mb-4 sm:mb-0">
          <Button 
            variant="ghost" 
            className="flex items-center space-x-2 text-gray-700 hover:text-pii-primary hover:bg-pii-light transition-colors"
            onClick={() => setLocationDialogOpen(true)}
          >
            <MapPin className="h-5 w-5" />
            <span>LOCATION</span>
          </Button>
          <Button 
            variant="ghost" 
            className="flex items-center space-x-2 text-gray-700 hover:text-pii-primary hover:bg-pii-light transition-colors"
            onClick={() => setContactDialogOpen(true)}
          >
            <Phone className="h-5 w-5" />
            <span>CONTACT US</span>
          </Button>
        </div>
        
        <div className="text-sm text-gray-500">
          © {new Date().getFullYear()} PII Shield. All rights reserved.
        </div>
      </div>

      <LocationDialog open={locationDialogOpen} onOpenChange={setLocationDialogOpen} />
      <ContactDialog open={contactDialogOpen} onOpenChange={setContactDialogOpen} />
    </footer>
  );
};

export default Footer;
