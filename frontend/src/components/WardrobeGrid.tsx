import { Plus, Loader } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ClothingItem {
  id: string;
  imageUrl: string;
  name: string;
}

interface WardrobeGridProps {
  items: ClothingItem[];
  onAddItem: () => void;
  isUploading?: boolean;
}

const WardrobeGrid = ({ items, onAddItem, isUploading = false }: WardrobeGridProps) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-charcoal">Your Wardrobe</h2>
          <p className="text-muted-foreground mt-1">{items.length} items in your collection</p>
        </div>
        <Button 
          onClick={onAddItem}
          disabled={isUploading}
          className="bg-primary text-primary-foreground hover:bg-primary/90 transition-colors duration-200 disabled:opacity-50"
        >
          {isUploading ? (
            <>
              <Loader className="w-4 h-4 mr-2 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <Plus className="w-4 h-4 mr-2" />
              Add Item
            </>
          )}
        </Button>
      </div>

      {items.length === 0 ? (
        <div className="text-center py-16 px-6">
          <div className="w-24 h-24 mx-auto mb-6 bg-accent rounded-full flex items-center justify-center">
            <Plus className="w-12 h-12 text-accent-foreground opacity-60" />
          </div>
          <h3 className="text-lg font-medium text-charcoal mb-2">Your wardrobe is empty</h3>
          <p className="text-muted-foreground mb-6">Upload your first clothing item to get started</p>
          <Button 
            onClick={onAddItem}
            size="lg"
            disabled={isUploading}
            className="bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {isUploading ? (
              <>
                <Loader className="w-4 h-4 mr-2 animate-spin" />
                Uploading...
              </>
            ) : (
              'Add Your First Item'
            )}
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {items.map((item, index) => (
            <div 
              key={item.id} 
              className="fashion-card group animate-fadeInUp"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="aspect-square relative overflow-hidden">
                <img
                  src={item.imageUrl}
                  alt={item.name}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-all duration-300"></div>
              </div>
              <div className="p-3">
                <h3 className="text-sm font-medium text-charcoal truncate">{item.name}</h3>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default WardrobeGrid;