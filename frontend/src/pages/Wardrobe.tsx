import { useState, useRef } from 'react';
import { ArrowLeft, Camera, User, Grid3X3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import WardrobeGrid from '@/components/WardrobeGrid';
import PhotoUpload from '@/components/PhotoUpload';
import { useToast } from '@/hooks/use-toast';

interface ClothingItem {
  id: string;
  imageUrl: string;
  name: string;
}

interface UserPhoto {
  id: string;
  originalUrl: string;
  processedUrl: string;
  name: string;
}

// Sample wardrobe items to demonstrate functionality
const initialWardrobe: ClothingItem[] = [
  {
    id: '1',
    imageUrl: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=400&fit=crop',
    name: 'White Cotton T-Shirt',
  },
  {
    id: '2', 
    imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=400&fit=crop',
    name: 'Dark Denim Jeans',
  },
  {
    id: '3',
    imageUrl: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400&h=400&fit=crop',
    name: 'White Sneakers',
  },
  {
    id: '4',
    imageUrl: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=400&fit=crop',
    name: 'Black Blazer',
  },
  {
    id: '5',
    imageUrl: 'https://images.unsplash.com/photo-1583743814966-8936f37f8302?w=400&h=400&fit=crop',
    name: 'Blue Dress Shirt',
  },
  {
    id: '6',
    imageUrl: 'https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400&h=400&fit=crop',
    name: 'Summer Dress',
  },
];

const Wardrobe = () => {
  const [wardrobeItems, setWardrobeItems] = useState<ClothingItem[]>(initialWardrobe);
  const [userPhotos, setUserPhotos] = useState<UserPhoto[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleAddItem = () => {
    fileInputRef.current?.click();
  };

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

    setIsUploading(true);
    
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);

      // Upload to backend API
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Upload failed');
      }

      // Create image URL from uploaded file for display
      const imageUrl = URL.createObjectURL(file);
      
      // Create new clothing item
      const newItem: ClothingItem = {
        id: `${Date.now()}`,
        imageUrl,
        name: file.name.split('.')[0] || `Clothing Item ${wardrobeItems.length + 1}`,
      };
      
      setWardrobeItems(prev => [...prev, newItem]);
      
      toast({
        title: "Item added!",
        description: `Your clothing item has been uploaded and saved. File: ${result.filename}`,
      });
      
    } catch (error) {
      console.error('Error uploading image:', error);
      toast({
        title: "Upload failed",
        description: `Could not upload the image: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const handlePhotosUpdate = (photos: UserPhoto[]) => {
    setUserPhotos(photos);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link to="/">
                <Button variant="ghost" size="sm" className="hover:bg-secondary">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to AI Generator
                </Button>
              </Link>
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 bg-primary rounded-md flex items-center justify-center">
                  <Camera className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-foreground">StyleSnap</h1>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 lg:px-8 py-12">
        <Tabs defaultValue="clothes" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="clothes" className="flex items-center gap-2">
              <Grid3X3 className="w-4 h-4" />
              Clothes ({wardrobeItems.length})
            </TabsTrigger>
            <TabsTrigger value="photos" className="flex items-center gap-2">
              <User className="w-4 h-4" />
              Photos ({userPhotos.length})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="clothes">
            <WardrobeGrid items={wardrobeItems} onAddItem={handleAddItem} isUploading={isUploading} />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </TabsContent>
          
          <TabsContent value="photos">
            <PhotoUpload photos={userPhotos} onPhotosUpdate={handlePhotosUpdate} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Wardrobe;