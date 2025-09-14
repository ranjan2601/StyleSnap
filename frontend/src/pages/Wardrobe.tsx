import { useState, useRef, useEffect } from 'react';
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
  tags?: any;
}

interface UserPhoto {
  id: string;
  originalUrl: string;
  processedUrl: string;
  name: string;
}

const Wardrobe = () => {
  const [wardrobeItems, setWardrobeItems] = useState<ClothingItem[]>([]);
  const [userPhotos, setUserPhotos] = useState<UserPhoto[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  // Function to fetch wardrobe items from backend
  const fetchWardrobeItems = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:7000/debug/wardrobe');
      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Failed to fetch wardrobe items');
      }

      // Transform backend data to frontend format
      const transformedItems: ClothingItem[] = result.items.map((item: any) => ({
        id: item.id.toString(),
        imageUrl: `http://localhost:7000/${item.image_path}`, // Use backend static file serving
        name: extractNameFromTags(item.tags) || `Item ${item.id}`,
        tags: item.tags
      }));

      setWardrobeItems(transformedItems);
    } catch (error) {
      console.error('Error fetching wardrobe items:', error);
      toast({
        title: "Failed to load wardrobe",
        description: `Could not fetch your wardrobe items: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to extract a meaningful name from tags
  const extractNameFromTags = (tags: any): string => {
    if (!tags) return '';
    
    // Try to get description first
    if (tags.description) return tags.description;
    
    // Otherwise build name from available tags
    const parts = [];
    if (tags.color?.value) parts.push(tags.color.value);
    if (tags.category?.value) parts.push(tags.category.value);
    if (tags.style?.value) parts.push(tags.style.value);
    
    return parts.join(' ') || '';
  };

  // Fetch wardrobe items on component mount
  useEffect(() => {
    fetchWardrobeItems();
  }, []);

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
      const response = await fetch('http://localhost:7000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Upload failed');
      }

      // Refresh wardrobe items from backend after successful upload
      await fetchWardrobeItems();
      
      toast({
        title: "Item added!",
        description: `Your clothing item has been uploaded and analyzed successfully.`,
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
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-muted-foreground">Loading your wardrobe...</p>
                </div>
              </div>
            ) : (
              <WardrobeGrid items={wardrobeItems} onAddItem={handleAddItem} isUploading={isUploading} />
            )}
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