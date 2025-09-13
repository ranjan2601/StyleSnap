import { useState } from 'react';
import { Upload, User, X, Loader, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { removeBackground, loadImage } from '@/utils/backgroundRemoval';

interface UserPhoto {
  id: string;
  originalUrl: string;
  processedUrl: string;
  name: string;
}

interface PhotoUploadProps {
  photos: UserPhoto[];
  onPhotosUpdate: (photos: UserPhoto[]) => void;
}

const PhotoUpload = ({ photos, onPhotosUpdate }: PhotoUploadProps) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file",
        description: "Please upload an image file.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    
    try {
      // Create original image URL
      const originalUrl = URL.createObjectURL(file);
      
      // Load image for processing
      const imageElement = await loadImage(file);
      
      // Remove background
      const processedBlob = await removeBackground(imageElement);
      const processedUrl = URL.createObjectURL(processedBlob);
      
      // Create new photo object
      const newPhoto: UserPhoto = {
        id: `${Date.now()}`,
        originalUrl,
        processedUrl,
        name: file.name.split('.')[0] || `Photo ${photos.length + 1}`,
      };
      
      // Update photos
      onPhotosUpdate([...photos, newPhoto]);
      
      toast({
        title: "Photo uploaded!",
        description: "Background removed and ready for virtual try-on.",
      });
      
    } catch (error) {
      console.error('Error processing image:', error);
      toast({
        title: "Processing failed",
        description: "Could not process the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const removePhoto = (photoId: string) => {
    const updatedPhotos = photos.filter(photo => {
      if (photo.id === photoId) {
        // Clean up object URLs
        URL.revokeObjectURL(photo.originalUrl);
        URL.revokeObjectURL(photo.processedUrl);
        return false;
      }
      return true;
    });
    onPhotosUpdate(updatedPhotos);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-charcoal">Your Photos</h2>
          <p className="text-muted-foreground mt-1">Upload photos to try clothes on virtually</p>
        </div>
        
        <div className="relative">
          <input
            type="file"
            id="photo-upload"
            accept="image/*"
            onChange={handleFileUpload}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isProcessing}
          />
          <Button 
            className="bg-primary text-primary-foreground hover:bg-luxury-gold-dark disabled:opacity-50"
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <Loader className="w-4 h-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload Photo
              </>
            )}
          </Button>
        </div>
      </div>

      {photos.length === 0 && !isProcessing ? (
        <div className="text-center py-16 px-6 border-2 border-dashed border-border rounded-xl">
          <div className="w-24 h-24 mx-auto mb-6 bg-accent rounded-full flex items-center justify-center">
            <User className="w-12 h-12 text-accent-foreground opacity-60" />
          </div>
          <h3 className="text-lg font-medium text-charcoal mb-2">No photos yet</h3>
          <p className="text-muted-foreground mb-6">Upload a photo to start virtual try-ons</p>
          <div className="relative inline-block">
            <input
              type="file"
              id="first-photo-upload"
              accept="image/*"
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Button 
              size="lg"
              className="bg-primary text-primary-foreground hover:bg-luxury-gold-dark"
            >
              Upload Your First Photo
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {photos.map((photo, index) => (
            <div 
              key={photo.id} 
              className="fashion-card group animate-fadeInUp"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="aspect-square relative overflow-hidden">
                <img
                  src={photo.processedUrl}
                  alt={photo.name}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-all duration-300"></div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => removePhoto(photo.id)}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 w-8 h-8 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
                <div className="absolute bottom-2 right-2 bg-luxury-gold text-white rounded-full p-1">
                  <Check className="w-3 h-3" />
                </div>
              </div>
              <div className="p-3">
                <h3 className="text-sm font-medium text-charcoal truncate">{photo.name}</h3>
                <p className="text-xs text-muted-foreground">Ready for try-on</p>
              </div>
            </div>
          ))}
          
          {isProcessing && (
            <div className="fashion-card">
              <div className="aspect-square relative overflow-hidden bg-muted flex items-center justify-center">
                <div className="text-center">
                  <Loader className="w-8 h-8 text-luxury-gold animate-spin mx-auto mb-2" />
                  <p className="text-xs text-muted-foreground">Processing...</p>
                </div>
              </div>
              <div className="p-3">
                <div className="h-4 bg-muted rounded animate-pulse"></div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PhotoUpload;