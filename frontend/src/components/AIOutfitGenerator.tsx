import { useState, useRef } from 'react';
import { Sparkles, Shuffle, X, CheckCircle2, MousePointerClick, Plus, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/hooks/use-toast';

interface ClothingItem {
  id: string;
  imageUrl: string;
  name: string;
}

interface UserPhotoLite {
  id: string;
  processedUrl: string;
  name: string;
}

interface AIOutfitGeneratorProps {
  wardrobeItems: ClothingItem[];
  userPhotos?: UserPhotoLite[];
  onAddClothingItem?: (item: ClothingItem) => void;
}

const AIOutfitGenerator = ({ wardrobeItems, userPhotos = [], onAddClothingItem }: AIOutfitGeneratorProps) => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedOutfit, setGeneratedOutfit] = useState<ClothingItem[]>([]);
  const [showOutfit, setShowOutfit] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([]);
  const [isApplying, setIsApplying] = useState(false);
  const [appliedPreviewUrl, setAppliedPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const generateOutfit = async () => {
    if (!prompt.trim() || wardrobeItems.length === 0) return;
    
    setIsGenerating(true);
    setCurrentPrompt(prompt);
    
    // Simulate AI processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Randomly select 2-4 items from wardrobe
    const shuffled = [...wardrobeItems].sort(() => 0.5 - Math.random());
    const outfitSize = Math.min(Math.floor(Math.random() * 3) + 2, wardrobeItems.length);
    const outfit = shuffled.slice(0, outfitSize);
    
    setGeneratedOutfit(outfit);
    setIsGenerating(false);
    setShowOutfit(true);
  };

  const regenerateOutfit = async () => {
    setIsGenerating(true);
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const shuffled = [...wardrobeItems].sort(() => 0.5 - Math.random());
    const outfitSize = Math.min(Math.floor(Math.random() * 3) + 2, wardrobeItems.length);
    const outfit = shuffled.slice(0, outfitSize);
    
    setGeneratedOutfit(outfit);
    setIsGenerating(false);
  };

  const closeOutfit = () => {
    setShowOutfit(false);
    setPrompt('');
    setSelectedItemIds([]);
    setAppliedPreviewUrl(null);
  };

  const loadHtmlImage = (src: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });

  const applySelectedToFirstPhoto = async () => {
    if (selectedItemIds.length === 0 || userPhotos.length === 0) return;
    setIsApplying(true);
    try {
      const basePhoto = userPhotos[0];
      const photoImg = await loadHtmlImage(basePhoto.processedUrl);

      const canvas = document.createElement('canvas');
      const maxW = 800;
      const scale = Math.min(1, maxW / photoImg.width);
      canvas.width = Math.floor(photoImg.width * scale);
      canvas.height = Math.floor(photoImg.height * scale);
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas not supported');

      ctx.drawImage(photoImg, 0, 0, canvas.width, canvas.height);

      // Overlay each selected item with slight vertical offsets
      const selectedItems = selectedItemIds
        .map(id => generatedOutfit.find(i => i.id === id))
        .filter((i): i is ClothingItem => Boolean(i));

      for (let i = 0; i < selectedItems.length; i++) {
        const item = selectedItems[i];
        const dressImg = await loadHtmlImage(item.imageUrl);
        const baseWidthRatio = 0.38; // ~38% of canvas width
        const overlayW = Math.floor(canvas.width * baseWidthRatio);
        const overlayScale = overlayW / dressImg.width;
        const overlayH = Math.floor(dressImg.height * overlayScale);
        const x = Math.floor((canvas.width - overlayW) / 2);
        const y = Math.floor(canvas.height * (0.22 + i * 0.12));
        ctx.globalAlpha = 0.9;
        ctx.drawImage(dressImg, x, y, overlayW, overlayH);
        ctx.globalAlpha = 1;
      }

      const url = canvas.toDataURL('image/png');
      setAppliedPreviewUrl(url);
    } catch (e) {
      console.error(e);
    } finally {
      setIsApplying(false);
    }
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
      const imageUrl = URL.createObjectURL(file);
      const newItem: ClothingItem = {
        id: `${Date.now()}`,
        imageUrl,
        name: file.name.split('.')[0] || `Clothing Item`,
      };
      
      onAddClothingItem?.(newItem);
      
      toast({
        title: "Item added!",
        description: "Your clothing item has been added to your wardrobe.",
      });
      
    } catch (error) {
      console.error('Error uploading image:', error);
      toast({
        title: "Upload failed",
        description: "Could not upload the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  return (
    <>
      <div className="space-y-4">
        {/* Example Chips - Above input like ChatGPT */}
        <div className="flex flex-wrap gap-2 justify-center">
          {['Casual Brunch', 'Formal Wedding', '80s Themed Party'].map((example) => (
              <button
                key={example}
                onClick={() => setPrompt(example)}
                className="px-4 py-2 bg-secondary hover:bg-accent/10 text-secondary-foreground hover:text-accent rounded-full text-sm font-light transition-colors duration-200 border border-border/50 hover:border-accent/30"
              >
                {example}
              </button>
          ))}
        </div>
        
        <div className="relative">
          <div className="bg-white border-2 border-border rounded-2xl py-6 shadow-sm hover:shadow-md transition-all duration-200 focus-within:border-primary/50 focus-within:shadow-lg flex items-center">
            {/* Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="ml-6 mr-4 p-2 hover:bg-secondary rounded-lg transition-colors duration-200 disabled:opacity-50"
              title="Upload clothing item"
            >
              {isUploading ? (
                <Upload className="w-5 h-5 text-muted-foreground animate-pulse" />
              ) : (
                <Plus className="w-5 h-5 text-muted-foreground" />
              )}
            </button>
            
            {/* Input */}
            <Input
              id="occasion"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What's the occasion?"
              className="flex-1 text-xl border-0 p-0 h-8 focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground mr-6"
              onKeyPress={(e) => e.key === 'Enter' && generateOutfit()}
            />
          </div>
          
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
        
        <Button 
          onClick={generateOutfit}
          disabled={!prompt.trim() || wardrobeItems.length === 0 || isGenerating}
          className="w-full h-12 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors duration-200 text-base font-medium"
          size="lg"
        >
          {isGenerating ? (
            <>
              <div className="outfit-shimmer w-4 h-4 rounded mr-2"></div>
              Generating Your Perfect Outfit...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 mr-2" />
              Generate Outfit
            </>
          )}
        </Button>
        
        {wardrobeItems.length === 0 && (
          <p className="text-sm text-muted-foreground text-center">
            Add some clothes to your wardrobe first!
          </p>
        )}
      </div>

      {/* Outfit Modal */}
      {showOutfit && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-background rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-auto animate-modalIn">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-xl font-semibold text-charcoal">Your Perfect Outfit</h3>
                  <p className="text-muted-foreground">For: "{currentPrompt}"</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={closeOutfit}
                  className="hover:bg-secondary"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>

              {isGenerating ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="aspect-square rounded-lg outfit-shimmer"></div>
                    ))}
                  </div>
                  <p className="text-center text-muted-foreground">Our AI is selecting the perfect pieces...</p>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {generatedOutfit.map((item, index) => {
                      const isSelected = selectedItemIds.includes(item.id);
                      return (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => {
                            setAppliedPreviewUrl(null);
                            setSelectedItemIds((prev) =>
                              prev.includes(item.id)
                                ? prev.filter(id => id !== item.id)
                                : [...prev, item.id]
                            );
                          }}
                          className={`fashion-card animate-fadeInUp text-left ${isSelected ? 'ring-2 ring-luxury-gold' : ''}`}
                          style={{ animationDelay: `${index * 0.15}s` }}
                        >
                          <div className="aspect-square relative overflow-hidden">
                            <img
                              src={item.imageUrl}
                              alt={item.name}
                              className="w-full h-full object-cover"
                            />
                            {isSelected && (
                              <div className="absolute top-2 right-2 bg-luxury-gold text-white rounded-full p-1">
                                <CheckCircle2 className="w-4 h-4" />
                              </div>
                            )}
                          </div>
                          <div className="p-3">
                            <h4 className="text-sm font-medium text-charcoal">{item.name}</h4>
                            {!isSelected && (
                              <div className="flex items-center gap-1 text-xs text-muted-foreground mt-1">
                                <MousePointerClick className="w-3 h-3" />
                                Select
                              </div>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  <div className="space-y-3">
                    <Button
                      onClick={applySelectedToFirstPhoto}
                      disabled={selectedItemIds.length === 0 || userPhotos.length === 0 || isApplying}
                      className="w-full bg-primary text-primary-foreground hover:bg-luxury-gold-dark disabled:opacity-50"
                    >
                      {isApplying
                        ? 'Applying…'
                        : userPhotos.length === 0
                        ? 'Upload a photo to apply'
                        : `Apply Selected (${selectedItemIds.length})`}
                    </Button>

                    {appliedPreviewUrl && (
                      <div className="border border-border/50 rounded-lg p-3">
                        <h5 className="text-sm font-medium text-charcoal mb-2">Preview</h5>
                        <img src={appliedPreviewUrl} alt="Applied preview" className="w-full rounded" />
                        <div className="mt-3 flex gap-3">
                          <a
                            href={appliedPreviewUrl}
                            download={`try-on-${Date.now()}.png`}
                            className="inline-flex items-center justify-center px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-luxury-gold-dark text-sm"
                          >
                            Download Image
                          </a>
                          <Button variant="outline" onClick={() => setAppliedPreviewUrl(null)} className="text-sm">
                            Clear Preview
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-3">
                    <Button 
                      onClick={regenerateOutfit}
                      variant="outline"
                      className="flex-1"
                      disabled={isGenerating}
                    >
                      <Shuffle className="w-4 h-4 mr-2" />
                      Try Again
                    </Button>
                    <Button 
                      onClick={closeOutfit}
                      className="flex-1 bg-primary text-primary-foreground hover:bg-luxury-gold-dark"
                    >
                      Perfect!
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default AIOutfitGenerator;