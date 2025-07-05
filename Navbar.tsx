
import React from 'react';
import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import ServiceDialog from './ServiceDialog';
import AboutDialog from './AboutDialog';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const [serviceDialogOpen, setServiceDialogOpen] = React.useState(false);
  const [aboutDialogOpen, setAboutDialogOpen] = React.useState(false);

  return (
    <nav className="w-full py-4 px-4 md:px-8 bg-white border-b border-gray-200">
      <div className="container mx-auto flex justify-between items-center">
        <div className="text-2xl md:text-3xl font-bold text-pii-primary">
          PII
        </div>
        
        {/* Mobile menu button */}
        <div className="md:hidden">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            aria-label="Toggle menu"
          >
            <Menu className="h-6 w-6" />
          </Button>
        </div>
        
        {/* Desktop navigation */}
        <div className="hidden md:flex space-x-4">
          <Button 
            variant="ghost" 
            className="text-gray-700 hover:text-pii-primary hover:bg-pii-light transition-colors"
            onClick={() => setServiceDialogOpen(true)}
          >
            SERVICES
          </Button>
          <Button 
            variant="ghost" 
            className="text-gray-700 hover:text-pii-primary hover:bg-pii-light transition-colors"
            onClick={() => setAboutDialogOpen(true)}
          >
            ABOUT US
          </Button>
        </div>
        
        {/* Mobile navigation dropdown */}
        {isMenuOpen && (
          <div className="absolute top-16 right-4 z-50 bg-white shadow-lg rounded-lg p-4 md:hidden animate-fade-in">
            <div className="flex flex-col space-y-2">
              <Button 
                variant="ghost"
                className="px-4 py-2 text-gray-700 hover:bg-pii-light hover:text-pii-primary rounded-md text-left justify-start"
                onClick={() => {
                  setServiceDialogOpen(true);
                  setIsMenuOpen(false);
                }}
              >
                SERVICES
              </Button>
              <Button 
                variant="ghost"
                className="px-4 py-2 text-gray-700 hover:bg-pii-light hover:text-pii-primary rounded-md text-left justify-start"
                onClick={() => {
                  setAboutDialogOpen(true);
                  setIsMenuOpen(false);
                }}
              >
                ABOUT US
              </Button>
            </div>
          </div>
        )}
      </div>

      <ServiceDialog open={serviceDialogOpen} onOpenChange={setServiceDialogOpen} />
      <AboutDialog open={aboutDialogOpen} onOpenChange={setAboutDialogOpen} />
    </nav>
  );
};

export default Navbar;
