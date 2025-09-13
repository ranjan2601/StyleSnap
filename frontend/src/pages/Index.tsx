import { useState } from 'react';
import { Camera, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import AIOutfitGenerator from '@/components/AIOutfitGenerator';

interface ClothingItem {
  id: string;
  imageUrl: string;
  name: string;
}

// Sample wardrobe items for AI generator
const sampleWardrobe: ClothingItem[] = [
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

interface UserPhotoLite {
  id: string;
  originalUrl?: string;
  processedUrl: string;
  name: string;
}

const Index = () => {
  const [userPhotos, setUserPhotos] = useState<UserPhotoLite[]>([]);
  const [wardrobeItems, setWardrobeItems] = useState<ClothingItem[]>(sampleWardrobe);
  
  const handleAddClothingItem = (item: ClothingItem) => {
    setWardrobeItems(prev => [...prev, item]);
  };
  
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border/20 bg-white/90 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 gold-accent rounded-xl flex items-center justify-center shadow-lg">
                <Camera className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl luxury-text text-foreground">StyleSnap</h1>
              </div>
            </div>
            <div className="flex items-center gap-8">
              <Link to="/wardrobe">
                <Button 
                  variant="outline"
                  className="luxury-button border-accent/30 hover:bg-accent/10 transition-all duration-300 font-medium tracking-wide"
                >
                  Your Wardrobe
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col justify-center">
        <div className="max-w-4xl mx-auto px-6 lg:px-8 py-16">
          <div className="text-center mb-20">
            <h1 className="text-5xl md:text-7xl font-bold text-foreground mb-8 tracking-tight leading-none">
              Your Personal AI Stylist
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto leading-relaxed font-light">
              Describe any occasion and let our AI create the perfect outfit from your wardrobe. 
              Never wonder what to wear again.
            </p>
          </div>
          
          {/* Centered Input */}
          <div className="max-w-4xl mx-auto">
            <AIOutfitGenerator 
              wardrobeItems={wardrobeItems} 
              userPhotos={userPhotos.map(p => ({ id: p.id, processedUrl: p.processedUrl, name: p.name }))}
              onAddClothingItem={handleAddClothingItem}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;

