/**
 * Main game engine for the lemonade stand business simulation
 * JavaScript port of the Python BusinessGame class
 */
class BusinessGame {
    constructor(days = 30, startingCash = 1000, seed = null) {
        // Game configuration
        this.totalDays = days;
        this.currentDay = 0;
        this.cash = startingCash;
        this.startingCash = startingCash;
        this.hourlyOperatingCost = 5.0;
        
        // Game state
        this.inventory = new Inventory();
        this.demandModel = new DemandModel();
        this.history = [];
        this.todaySupplyCosts = {};
        this.yesterdayProfit = null;
        
        // Daily settings (reset each day)
        this.price = null;
        this.openHour = null;
        this.closeHour = null;
        this.priceSet = false;
        this.hoursSet = false;
        
        // Random seed for reproducible games
        if (seed !== null) {
            this.demandModel.setRandomSeed(seed);
            this.setSeed(seed);
        }
        
        // Track supply cost history for analysis
        this.supplyCostHistory = [];
    }

    /**
     * Set random seed for reproducible gameplay
     * @param {number} seed - Random seed value
     */
    setSeed(seed) {
        this.seed = seed;
        this._rng = new SeededRandom(seed);
    }

    /**
     * Start a new day and return day information
     * @returns {Object} Day start information
     */
    startNewDay() {
        this.currentDay += 1;
        
        // Remove expired inventory
        const expiredItems = this.inventory.removeExpired(this.currentDay);
        
        // Generate today's supply costs (±10% variation)
        this.todaySupplyCosts = this.generateSupplyCosts();
        
        // Reset daily settings
        this.price = null;
        this.openHour = null;
        this.closeHour = null;
        this.priceSet = false;
        this.hoursSet = false;
        
        const dayInfo = {
            day: this.currentDay,
            cash: this.cash,
            expiredItems: expiredItems,
            supplyCosts: { ...this.todaySupplyCosts },
            inventorySummary: this.inventory.getSummary()
        };
        
        return dayInfo;
    }

    /**
     * Generate randomized supply costs for the day
     * @returns {Object} Supply costs for each item type
     */
    generateSupplyCosts() {
        const costs = {};
        const baseCosts = this.inventory.baseCosts;
        
        for (const [itemType, baseCost] of Object.entries(baseCosts)) {
            // ±10% variation
            const variation = this._rng ? this._rng.random() : Math.random();
            const factor = 0.9 + (variation * 0.2); // 0.9 to 1.1 range
            costs[itemType] = Math.round(baseCost * factor * 100) / 100;
        }
        
        return costs;
    }

    /**
     * Check morning supply prices
     * @returns {Object} Result with success flag and prices
     */
    checkMorningPrices() {
        return {
            success: true,
            prices: { ...this.todaySupplyCosts }
        };
    }

    /**
     * Order supplies if player has sufficient funds
     * @param {Object} order - Order quantities {cups, lemons, sugar, water}
     * @returns {Object} Result with success flag and details
     */
    orderSupplies(order = {}) {
        const cups = order.cups || 0;
        const lemons = order.lemons || 0;
        const sugar = order.sugar || 0;
        const water = order.water || 0;

        // Calculate total cost
        const totalCost = 
            (cups * this.todaySupplyCosts.cups) +
            (lemons * this.todaySupplyCosts.lemons) +
            (sugar * this.todaySupplyCosts.sugar) +
            (water * this.todaySupplyCosts.water);

        if (totalCost > this.cash) {
            return {
                success: false,
                error: `Insufficient funds. Need $${totalCost.toFixed(2)}, have $${this.cash.toFixed(2)}`,
                totalCost: totalCost,
                remainingCash: this.cash
            };
        }

        // Deduct cost and add items to inventory
        this.cash -= totalCost;
        
        if (cups > 0) this.inventory.addItems('cups', cups, this.currentDay);
        if (lemons > 0) this.inventory.addItems('lemons', lemons, this.currentDay);
        if (sugar > 0) this.inventory.addItems('sugar', sugar, this.currentDay);
        if (water > 0) this.inventory.addItems('water', water, this.currentDay);

        return {
            success: true,
            totalCost: totalCost,
            remainingCash: this.cash,
            inventorySummary: this.inventory.getSummary()
        };
    }

    /**
     * Set operating hours for the day
     * @param {number} openHour - Opening hour (0-23)
     * @param {number} closeHour - Closing hour (0-23)
     * @returns {Object} Result with success flag
     */
    setOperatingHours(openHour, closeHour) {
        if (openHour < 0 || openHour > 23) {
            return {
                success: false,
                error: `Invalid open hour: ${openHour}. Must be 0-23.`
            };
        }

        if (closeHour < 0 || closeHour > 23) {
            return {
                success: false,
                error: `Invalid close hour: ${closeHour}. Must be 0-23.`
            };
        }

        if (closeHour <= openHour) {
            return {
                success: false,
                error: `Close hour (${closeHour}) must be after open hour (${openHour}).`
            };
        }

        this.openHour = openHour;
        this.closeHour = closeHour;
        this.hoursSet = true;

        const hoursOpen = closeHour - openHour;
        const operatingCost = hoursOpen * this.hourlyOperatingCost;

        return {
            success: true,
            openHour: openHour,
            closeHour: closeHour,
            hoursOpen: hoursOpen,
            operatingCost: operatingCost
        };
    }

    /**
     * Set lemonade price for the day
     * @param {number} price - Price per lemonade
     * @returns {Object} Result with success flag
     */
    setPrice(price) {
        if (price < 0) {
            return {
                success: false,
                error: `Price cannot be negative: $${price}`
            };
        }

        this.price = price;
        this.priceSet = true;

        return {
            success: true,
            price: price,
            expectedDemand: this.hoursSet ? 
                this.demandModel.getExpectedDailyDemand(price, this.openHour, this.closeHour) : null
        };
    }

    /**
     * Check if ready to simulate the day
     * @returns {Array} [isReady, missingRequirements]
     */
    checkReadyForNextDay() {
        const missing = [];
        
        if (!this.priceSet) {
            missing.push("Price not set");
        }
        
        if (!this.hoursSet) {
            missing.push("Operating hours not set");
        }
        
        return [missing.length === 0, missing];
    }

    /**
     * Simulate a full business day
     * @returns {Object} Day simulation results
     */
    simulateDay() {
        const [ready, missing] = this.checkReadyForNextDay();
        
        if (!ready) {
            return {
                success: false,
                error: `Cannot simulate day: ${missing.join(', ')}`
            };
        }

        const hoursOpen = this.closeHour - this.openHour;
        const operatingCost = hoursOpen * this.hourlyOperatingCost;
        
        // Get hourly customer demand
        const hourlyCustomers = this.demandModel.calculateDailyCustomers(
            this.price, this.openHour, this.closeHour, true
        );
        
        // Simulate serving customers hour by hour
        let totalCustomersServed = 0;
        let totalCustomersLost = 0;
        const hourlySales = {};
        const recipe = { cups: 1, lemons: 1, sugar: 1, water: 1 };
        
        for (const [hour, customers] of Object.entries(hourlyCustomers)) {
            let served = 0;
            
            for (let i = 0; i < customers; i++) {
                if (this.inventory.useItems(recipe)) {
                    served++;
                } else {
                    // Out of stock - can't serve more customers
                    break;
                }
            }
            
            hourlySales[hour] = {
                demand: customers,
                served: served,
                lost: customers - served
            };
            
            totalCustomersServed += served;
            totalCustomersLost += (customers - served);
        }
        
        // Calculate financial results
        const revenue = totalCustomersServed * this.price;
        const profit = revenue - operatingCost;
        
        // Update cash and track profit
        this.cash += profit;
        this.yesterdayProfit = profit;
        
        // Record day in history
        const dayResult = {
            day: this.currentDay,
            price: this.price,
            openHour: this.openHour,
            closeHour: this.closeHour,
            hoursOpen: hoursOpen,
            customersServed: totalCustomersServed,
            customersLost: totalCustomersLost,
            revenue: Math.round(revenue * 100) / 100,
            operatingCost: operatingCost,
            profit: Math.round(profit * 100) / 100,
            hourlySales: hourlySales,
            endingCash: Math.round(this.cash * 100) / 100,
            inventoryAfter: this.inventory.getSummary()
        };
        
        this.history.push(dayResult);
        
        // Store supply costs in history
        this.supplyCostHistory.push({
            day: this.currentDay,
            ...this.todaySupplyCosts
        });
        
        return dayResult;
    }

    /**
     * Check if game is over (bankrupt or completed all days)
     * @returns {boolean} True if game is over
     */
    isGameOver() {
        return this.cash < 0 || this.currentDay >= this.totalDays;
    }

    /**
     * Get final game results and statistics
     * @returns {Object} Final game results
     */
    getFinalResults() {
        const totalProfit = this.cash - this.startingCash;
        const totalCustomers = this.history.reduce((sum, day) => sum + day.customersServed, 0);
        const totalRevenue = this.history.reduce((sum, day) => sum + day.revenue, 0);
        const totalOperatingCost = this.history.reduce((sum, day) => sum + day.operatingCost, 0);
        
        const averageDailyProfit = this.history.length > 0 ? totalProfit / this.history.length : 0;
        const averagePrice = this.history.length > 0 ? 
            this.history.reduce((sum, day) => sum + day.price, 0) / this.history.length : 0;
        
        return {
            daysPlayed: this.currentDay,
            finalCash: Math.round(this.cash * 100) / 100,
            totalProfit: Math.round(totalProfit * 100) / 100,
            totalCustomers: totalCustomers,
            totalRevenue: Math.round(totalRevenue * 100) / 100,
            totalOperatingCost: totalOperatingCost,
            averageDailyProfit: Math.round(averageDailyProfit * 100) / 100,
            averagePrice: Math.round(averagePrice * 100) / 100,
            inventoryValue: this.inventory.getTotalValue(),
            isBankrupt: this.cash < 0,
            completedAllDays: this.currentDay >= this.totalDays
        };
    }

    /**
     * Get turn prompt for current game state
     * @returns {string} Prompt describing current situation
     */
    getTurnPrompt() {
        if (this.currentDay === 0) {
            return `Welcome to LemonadeBench! You have $${this.cash.toFixed(2)} to run a profitable lemonade stand for ${this.totalDays} days. 

Click "Start New Day" to begin Day 1. Each day you'll:
1. Check supply prices and order ingredients 
2. Set your lemonade price and operating hours
3. Open for business and serve customers!

Your goal is to maximize profit. The optimal price is around $2.50 based on the demand curve: Q = 50 - 10p

Good luck! 🍋`;
        }

        let prompt = `Day ${this.currentDay} of ${this.totalDays}\n\n`;
        
        if (this.yesterdayProfit !== null) {
            const profitText = this.yesterdayProfit >= 0 ? 
                `made $${this.yesterdayProfit.toFixed(2)}` : 
                `lost $${Math.abs(this.yesterdayProfit).toFixed(2)}`;
            prompt += `Yesterday you ${profitText} in profit.\n\n`;
        }
        
        prompt += `Current cash: $${this.cash.toFixed(2)}\n`;
        const summary = this.inventory.getSummary();
        prompt += `Inventory: ${summary.cups} cups, ${summary.lemons} lemons, ${summary.sugar} sugar, ${summary.water} water\n`;
        prompt += `Can make: ${summary.canMake} lemonades\n\n`;
        
        if (this.isGameOver()) {
            if (this.cash < 0) {
                prompt += "GAME OVER - You're bankrupt! 💸";
            } else {
                prompt += "GAME COMPLETE! Time to see how you did! 🎉";
            }
        }
        
        return prompt;
    }

    /**
     * Get historical supply costs
     * @returns {Array} Array of daily supply costs
     */
    getHistoricalSupplyCosts() {
        return [...this.supplyCostHistory];
    }

    /**
     * Reset game to initial state
     */
    reset() {
        this.currentDay = 0;
        this.cash = this.startingCash;
        this.inventory = new Inventory();
        this.history = [];
        this.supplyCostHistory = [];
        this.todaySupplyCosts = {};
        this.yesterdayProfit = null;
        this.price = null;
        this.openHour = null;
        this.closeHour = null;
        this.priceSet = false;
        this.hoursSet = false;
    }
}