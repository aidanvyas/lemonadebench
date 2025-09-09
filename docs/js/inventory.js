/**
 * Inventory management system with FIFO expiration tracking
 * JavaScript port of the Python Inventory class
 */
class Inventory {
    constructor() {
        // Store items as arrays of {quantity, expiryDay} objects
        this.items = {
            cups: [],
            lemons: [],
            sugar: [],
            water: []
        };

        // Shelf life in days for each item type
        this.shelfLife = {
            cups: 30,
            lemons: 7,
            sugar: 60,
            water: Infinity  // Water never expires
        };

        // Base costs for reference (actual costs vary daily)
        this.baseCosts = {
            cups: 0.05,
            lemons: 0.20,
            sugar: 0.10,
            water: 0.02
        };
    }

    /**
     * Add items to inventory with expiration date
     * @param {string} itemType - Type of item ('cups', 'lemons', 'sugar', 'water')
     * @param {number} quantity - Number of items to add
     * @param {number} currentDay - Current day number for calculating expiry
     */
    addItems(itemType, quantity, currentDay) {
        if (!(itemType in this.items)) {
            throw new Error(`Unknown item type: ${itemType}`);
        }

        if (quantity <= 0) {
            return;
        }

        // Calculate expiry day (infinite for water)
        const expiryDay = this.shelfLife[itemType] === Infinity ? 
            Infinity : currentDay + this.shelfLife[itemType];

        // Add to inventory queue
        this.items[itemType].push({ quantity, expiryDay });
    }

    /**
     * Get total available quantity of an item type
     * @param {string} itemType - Type of item to check
     * @returns {number} Total quantity available
     */
    getAvailable(itemType) {
        if (!(itemType in this.items)) {
            return 0;
        }

        return this.items[itemType].reduce((total, batch) => total + batch.quantity, 0);
    }

    /**
     * Get detailed inventory information including expiration dates
     * @returns {Object} Dictionary with item types as keys and list of batches as values
     */
    getInventoryDetails() {
        const details = {};
        
        for (const [itemType, batches] of Object.entries(this.items)) {
            details[itemType] = batches.map(batch => ({
                quantity: batch.quantity,
                expiresDay: batch.expiryDay === Infinity ? "never" : batch.expiryDay
            }));
        }
        
        return details;
    }

    /**
     * Use items according to recipe, FIFO style
     * @param {Object} recipe - Dictionary of itemType -> quantity needed
     * @returns {boolean} True if all items were available and used, false otherwise
     */
    useItems(recipe) {
        // First check if we have enough of everything
        for (const [itemType, needed] of Object.entries(recipe)) {
            if (this.getAvailable(itemType) < needed) {
                return false;
            }
        }

        // Use items FIFO style (oldest first)
        for (const [itemType, needed] of Object.entries(recipe)) {
            let remaining = needed;
            const batches = this.items[itemType];
            
            for (let i = 0; i < batches.length && remaining > 0; i++) {
                const batch = batches[i];
                const takeFromBatch = Math.min(batch.quantity, remaining);
                
                batch.quantity -= takeFromBatch;
                remaining -= takeFromBatch;
                
                // Remove empty batches
                if (batch.quantity === 0) {
                    batches.splice(i, 1);
                    i--; // Adjust index after removal
                }
            }
        }
        
        return true;
    }

    /**
     * Remove expired items and return what was lost
     * @param {number} currentDay - Current day to check expiration against
     * @returns {Object} Dictionary of expired items by type
     */
    removeExpired(currentDay) {
        const expired = {};
        
        for (const [itemType, batches] of Object.entries(this.items)) {
            let expiredCount = 0;
            
            // Remove expired batches (keep non-expired ones)
            this.items[itemType] = batches.filter(batch => {
                if (batch.expiryDay !== Infinity && currentDay >= batch.expiryDay) {
                    expiredCount += batch.quantity;
                    return false; // Remove this batch
                }
                return true; // Keep this batch
            });
            
            if (expiredCount > 0) {
                expired[itemType] = expiredCount;
            }
        }
        
        return expired;
    }

    /**
     * Calculate how many lemonades can be made with current inventory
     * @returns {number} Maximum number of lemonades that can be made
     */
    canMakeLemonade() {
        const recipe = { cups: 1, lemons: 1, sugar: 1, water: 1 };
        
        return Math.min(
            this.getAvailable('cups'),
            this.getAvailable('lemons'),
            this.getAvailable('sugar'),
            this.getAvailable('water')
        );
    }

    /**
     * Calculate total inventory value based on base costs
     * @returns {number} Total value of all inventory
     */
    getTotalValue() {
        let total = 0;
        
        for (const [itemType, batches] of Object.entries(this.items)) {
            const quantity = batches.reduce((sum, batch) => sum + batch.quantity, 0);
            total += quantity * this.baseCosts[itemType];
        }
        
        return Math.round(total * 100) / 100; // Round to 2 decimal places
    }

    /**
     * Get a summary of current inventory for UI display
     * @returns {Object} Inventory summary with counts and can-make info
     */
    getSummary() {
        return {
            cups: this.getAvailable('cups'),
            lemons: this.getAvailable('lemons'),
            sugar: this.getAvailable('sugar'),
            water: this.getAvailable('water'),
            canMake: this.canMakeLemonade(),
            totalValue: this.getTotalValue()
        };
    }
}