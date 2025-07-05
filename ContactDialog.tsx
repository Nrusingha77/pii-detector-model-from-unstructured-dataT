
import React from 'react';
import { Mail, Send } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface ContactDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ContactDialog = ({ open, onOpenChange }: ContactDialogProps) => {
  const { toast } = useToast();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Message sent",
      description: "Thanks for reaching out! We'll get back to you soon.",
    });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-3xl font-bold text-pii-dark">Contact Us</DialogTitle>
          <DialogDescription className="text-gray-600">
            Have questions or feedback? We'd love to hear from you!
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-6 space-y-6">
          <div className="bg-pii-light p-6 rounded-lg">
            <h3 className="text-lg font-semibold mb-4 text-pii-dark flex items-center">
              <Mail className="h-5 w-5 mr-2 text-pii-primary" />
              Email Us
            </h3>
            <ul className="space-y-3 text-gray-700">
              <li>
                <a href="mailto:22cseaiml009.amanraj@giet.edu" className="hover:text-pii-primary transition-colors">
                  22cseaiml009.amanraj@giet.edu
                </a>
              </li>
              <li>
                <a href="mailto:22cseaiml011.nrusinghaprasadkhadanga@giet.edu" className="hover:text-pii-primary transition-colors">
                  22cseaiml011.nrusinghaprasadkhadanga@giet.edu
                </a>
              </li>
              <li>
                <a href="mailto:22cseaiml030.mukundarajasaha@giet.edu" className="hover:text-pii-primary transition-colors">
                  22cseaiml030.mukundarajasaha@giet.edu
                </a>
              </li>
            </ul>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded-lg shadow-md">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-pii-primary focus:border-pii-primary"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-pii-primary focus:border-pii-primary"
                  required
                />
              </div>
            </div>
            
            <div>
              <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-1">
                Subject
              </label>
              <input
                type="text"
                id="subject"
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-pii-primary focus:border-pii-primary"
                required
              />
            </div>
            
            <div>
              <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">
                Message
              </label>
              <textarea
                id="message"
                rows={5}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-pii-primary focus:border-pii-primary"
                required
              ></textarea>
            </div>
            
            <div>
              <Button 
                type="submit" 
                className="w-full bg-pii-primary hover:bg-pii-secondary flex items-center justify-center gap-2"
              >
                <Send className="h-4 w-4" />
                Send Message
              </Button>
            </div>
          </form>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ContactDialog;
