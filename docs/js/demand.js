/**
 * Demand modeling system for customer calculations
 * JavaScript port of the Python DemandModel class
 */
class DemandModel {
    // Hourly demand multipliers for all 24 hours
    static HOURLY_MULTIPLIERS = {
        0: 0.0,   // 12-1am: Closed
        1: 0.0,   // 1-2am: Closed
        2: 0.0,   // 2-3am: Closed
        3: 0.0,   // 3-4am: Closed
        4: 0.0,   // 4-5am: Closed
        5: 0.0,   // 5-6am: Closed
        6: 0.3,   // 6-7am: Early morning (30% of base)
        7: 0.5,   // 7-8am: Morning commute
        8: 0.7,   // 8-9am: Morning
        9: 0.8,   // 9-10am: Mid-morning
        10: 1.0,  // 10-11am: Late morning (100% base)
        11: 1.2,  // 11am-12pm: Pre-lunch
        12: 1.5,  // 12-1pm: Lunch peak (150% of base)
        13: 1.3,  // 1-2pm: Post-lunch
        14: 0.9,  // 2-3pm: Afternoon
        15: 0.8,  // 3-4pm: Mid-afternoon
        16: 0.9,  // 4-5pm: Late afternoon
        17: 1.1,  // 5-6pm: Evening commute
        18: 1.0,  // 6-7pm: Early evening
        19: 0.7,  // 7-8pm: Evening
        20: 0.4,  // 8-9pm: Late evening (40% of base)
        21: 0.0,  // 9-10pm: Closed
        22: 0.0,  // 10-11pm: Closed
        23: 0.0   // 11pm-12am: Closed
    };

    constructor(baseDemandIntercept = 50, priceSensitivity = 10) {
        /**
         * Initialize demand model
         * @param {number} baseDemandIntercept - Maximum customers per hour at price=0
         * @param {number} priceSensitivity - How much demand decreases per dollar of price
         */
        this.baseDemandIntercept = baseDemandIntercept;
        this.priceSensitivity = priceSensitivity;
        this._rng = null; // For random seed support
    }

    /**
     * Set random seed for reproducible simulations
     * @param {number} seed - Random seed value
     */
    setRandomSeed(seed) {
        // Simple seedable random number generator
        if (typeof SeededRandom === 'undefined') {
            console.warn('SeededRandom class not found, using Math.random()');
            this._rng = null;
            return;
        }
        this._rng = new SeededRandom(seed);
    }

    /**
     * Calculate base hourly demand at given price
     * Uses linear demand curve: demand = intercept - sensitivity * price
     * @param {number} price - Price per lemonade
     * @returns {number} Base demand (before time-of-day and random adjustments)
     */
    calculateBaseDemand(price) {
        const demand = this.baseDemandIntercept - this.priceSensitivity * price;
        return Math.max(0, demand); // Demand can't be negative
    }

    /**
     * Get demand multiplier for given hour
     * @param {number} hour - Hour of day (0-23)
     * @returns {number} Multiplier value (0.0 means closed)
     */
    getHourMultiplier(hour) {
        return DemandModel.HOURLY_MULTIPLIERS[hour] || 0.0;
    }

    /**
     * Calculate actual number of customers for a given hour
     * @param {number} price - Price per lemonade
     * @param {number} hour - Hour of day (0-23)
     * @param {boolean} randomVariation - Whether to apply ±10% random variation
     * @returns {number} Number of customers (rounded to nearest integer)
     */
    calculateCustomers(price, hour, randomVariation = true) {
        // Validate inputs
        if (typeof price !== 'number' || isNaN(price)) {
            console.error('🔧 Invalid price in calculateCustomers:', price);
            return 0;
        }
        
        if (typeof hour !== 'number' || isNaN(hour) || hour < 0 || hour > 23) {
            console.error('🔧 Invalid hour in calculateCustomers:', hour);
            return 0;
        }
        
        // Get base demand from price
        const baseDemand = this.calculateBaseDemand(price);

        // Apply time-of-day multiplier
        const hourMultiplier = this.getHourMultiplier(hour);
        let demandWithTime = baseDemand * hourMultiplier;

        // Apply random variation (±10%)
        if (randomVariation && demandWithTime > 0) {
            const variation = this._rng ? this._rng.random() : Math.random();
            const factor = 0.9 + (variation * 0.2); // 0.9 to 1.1 range
            demandWithTime *= factor;
        }

        const result = Math.round(demandWithTime);
        
        // Debug extremely high results
        if (result > 1000) {
            console.warn('🔧 Very high customer demand:', {
                price,
                hour,
                baseDemand,
                hourMultiplier,
                demandWithTime,
                result
            });
        }

        return result;
    }

    /**
     * Calculate customers for full day across operating hours
     * @param {number} price - Price per lemonade
     * @param {number} openHour - Opening hour (0-23)
     * @param {number} closeHour - Closing hour (0-23)
     * @param {boolean} randomVariation - Whether to apply random variation
     * @returns {Object} Dictionary of hour -> customer count
     */
    calculateDailyCustomers(price, openHour, closeHour, randomVariation = true) {
        const customers = {};
        
        for (let hour = openHour; hour < closeHour; hour++) {
            customers[hour] = this.calculateCustomers(price, hour, randomVariation);
        }
        
        return customers;
    }

    /**
     * Get expected demand preview for UI (no randomness)
     * @param {number} price - Price per lemonade
     * @param {number} openHour - Opening hour
     * @param {number} closeHour - Closing hour
     * @returns {number} Total expected customers for the day
     */
    getExpectedDailyDemand(price, openHour, closeHour) {
        let totalDemand = 0;
        
        for (let hour = openHour; hour < closeHour; hour++) {
            totalDemand += this.calculateCustomers(price, hour, false);
        }
        
        return totalDemand;
    }
}