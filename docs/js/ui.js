/**
 * UI Controller for the LemonadeBench web game
 * Manages all game interactions and interface updates
 */
class GameUI {
    constructor() {
        this.game = null;
        this.currentPhase = 'pre-game'; // pre-game, morning, planning, results, game-over
        
        // Initialize game
        this.initializeGame();
        
        // Bind event handlers
        this.bindEventHandlers();
        
        // Update initial display
        this.updateDisplay();
    }

    /**
     * Initialize new game instance
     */
    initializeGame() {
        this.game = new BusinessGame(30, 1000, Math.floor(Math.random() * 10000));
        this.currentPhase = 'pre-game';
        this.updateDisplay();
    }

    /**
     * Bind all event handlers
     */
    bindEventHandlers() {
        // Main action buttons
        document.getElementById('start-day-btn').addEventListener('click', () => {
            this.startNewDay();
        });

        document.getElementById('next-phase-btn').addEventListener('click', () => {
            this.nextPhase();
        });

        document.getElementById('restart-btn').addEventListener('click', () => {
            this.restartGame();
        });

        // Order form inputs - update costs dynamically
        const orderInputs = ['order-cups', 'order-lemons', 'order-sugar', 'order-water'];
        orderInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('input', () => this.updateOrderTotal());
            }
        });

        // Planning form inputs - update previews
        const priceInput = document.getElementById('lemonade-price');
        if (priceInput) {
            priceInput.addEventListener('input', () => this.updateDemandPreview());
        }

        const hourInputs = ['open-hour', 'close-hour'];
        hourInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('change', () => this.updateOperatingCostPreview());
            }
        });
    }

    /**
     * Start a new day
     */
    startNewDay() {
        const dayInfo = this.game.startNewDay();
        this.currentPhase = 'morning';
        this.updateDisplay();
        this.showMorningPhase(dayInfo);
    }

    /**
     * Show morning phase - supply ordering
     */
    showMorningPhase(dayInfo) {
        const gamePhase = document.getElementById('game-phase');
        const template = document.getElementById('morning-phase-template');
        gamePhase.innerHTML = template.innerHTML;

        // Show supply prices
        this.displaySupplyPrices(dayInfo.supplyCosts);

        // Rebind event handlers for new elements
        this.bindMorningEventHandlers();
        
        // Show expired items alert if any
        if (Object.keys(dayInfo.expiredItems).length > 0) {
            this.showExpiredItemsAlert(dayInfo.expiredItems);
        }
    }

    /**
     * Display today's supply prices
     */
    displaySupplyPrices(prices) {
        const pricesDiv = document.getElementById('supply-prices');
        const items = [
            { type: 'cups', icon: '🥤', name: 'Cups' },
            { type: 'lemons', icon: '🍋', name: 'Lemons' },
            { type: 'sugar', icon: '🍯', name: 'Sugar' },
            { type: 'water', icon: '💧', name: 'Water' }
        ];

        pricesDiv.innerHTML = items.map(item => `
            <div class="price-item">
                <div class="item-icon">${item.icon}</div>
                <div class="item-name">${item.name}</div>
                <div class="item-price">$${prices[item.type].toFixed(2)}</div>
            </div>
        `).join('');
    }

    /**
     * Bind event handlers for morning phase
     */
    bindMorningEventHandlers() {
        document.getElementById('place-order-btn').addEventListener('click', () => {
            this.placeOrder();
        });

        // Update order costs as user types
        const orderInputs = ['order-cups', 'order-lemons', 'order-sugar', 'order-water'];
        orderInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            input.addEventListener('input', () => this.updateOrderTotal());
        });

        // Initialize order total
        this.updateOrderTotal();
    }

    /**
     * Update order total cost display
     */
    updateOrderTotal() {
        const prices = this.game.todaySupplyCosts;
        const items = ['cups', 'lemons', 'sugar', 'water'];
        let total = 0;

        items.forEach(item => {
            const input = document.getElementById(`order-${item}`);
            const quantity = parseInt(input.value) || 0;
            const cost = quantity * prices[item];
            total += cost;

            // Update individual cost displays
            const costDisplay = input.parentElement.querySelector('.cost-display');
            costDisplay.textContent = `$${cost.toFixed(2)}`;
        });

        document.getElementById('total-order-cost').textContent = total.toFixed(2);
    }

    /**
     * Place supply order
     */
    placeOrder() {
        const order = {
            cups: parseInt(document.getElementById('order-cups').value) || 0,
            lemons: parseInt(document.getElementById('order-lemons').value) || 0,
            sugar: parseInt(document.getElementById('order-sugar').value) || 0,
            water: parseInt(document.getElementById('order-water').value) || 0
        };

        const result = this.game.orderSupplies(order);

        if (result.success) {
            this.updateDisplay();
            this.currentPhase = 'planning';
            this.showPlanningPhase();
        } else {
            alert(result.error);
        }
    }

    /**
     * Show planning phase - set price and hours
     */
    showPlanningPhase() {
        const gamePhase = document.getElementById('game-phase');
        const template = document.getElementById('planning-phase-template');
        gamePhase.innerHTML = template.innerHTML;

        this.bindPlanningEventHandlers();
    }

    /**
     * Bind event handlers for planning phase
     */
    bindPlanningEventHandlers() {
        document.getElementById('open-for-business-btn').addEventListener('click', () => {
            this.openForBusiness();
        });

        document.getElementById('lemonade-price').addEventListener('input', () => {
            this.updateDemandPreview();
        });

        const hourInputs = ['open-hour', 'close-hour'];
        hourInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            input.addEventListener('change', () => this.updateOperatingCostPreview());
        });

        // Initialize previews
        this.updateDemandPreview();
        this.updateOperatingCostPreview();
    }

    /**
     * Update demand preview based on current price and hours
     */
    updateDemandPreview() {
        const priceInput = document.getElementById('lemonade-price');
        const price = parseFloat(priceInput.value);
        const demandPreview = document.getElementById('demand-preview');

        if (price && price >= 0) {
            const openHour = parseInt(document.getElementById('open-hour').value);
            const closeHour = parseInt(document.getElementById('close-hour').value);
            const expectedDemand = this.game.demandModel.getExpectedDailyDemand(price, openHour, closeHour);
            demandPreview.textContent = `${expectedDemand} cups`;
        } else {
            demandPreview.textContent = '-- cups';
        }
    }

    /**
     * Update operating cost preview based on selected hours
     */
    updateOperatingCostPreview() {
        const openHour = parseInt(document.getElementById('open-hour').value);
        const closeHour = parseInt(document.getElementById('close-hour').value);
        const hoursOpen = closeHour - openHour;
        const cost = hoursOpen * this.game.hourlyOperatingCost;

        document.getElementById('operating-cost-preview').textContent = cost.toFixed(2);
    }

    /**
     * Open for business - set price, hours, and simulate day
     */
    openForBusiness() {
        try {
            console.log('🔧 Debug: openForBusiness() called');
            
            const priceElement = document.getElementById('lemonade-price');
            const openHourElement = document.getElementById('open-hour');
            const closeHourElement = document.getElementById('close-hour');
            
            console.log('🔧 Debug: Elements found:', {
                priceElement: !!priceElement,
                openHourElement: !!openHourElement,
                closeHourElement: !!closeHourElement
            });
            
            if (!priceElement || !openHourElement || !closeHourElement) {
                alert('Error: Could not find form elements. Please refresh and try again.');
                return;
            }
            
            const price = parseFloat(priceElement.value);
            const openHour = parseInt(openHourElement.value);
            const closeHour = parseInt(closeHourElement.value);
            
            console.log('🔧 Debug: Values:', { price, openHour, closeHour });

            // Validate inputs
            if (!price || price < 0) {
                alert('Please enter a valid price');
                return;
            }

            // Set price and hours
            console.log('🔧 Debug: Setting price and hours...');
            const priceResult = this.game.setPrice(price);
            const hoursResult = this.game.setOperatingHours(openHour, closeHour);
            
            console.log('🔧 Debug: Results:', { priceResult, hoursResult });

            if (!priceResult.success) {
                alert(priceResult.error);
                return;
            }

            if (!hoursResult.success) {
                alert(hoursResult.error);
                return;
            }

            // Simulate the day
            console.log('🔧 Debug: Simulating day...');
            const dayResult = this.game.simulateDay();
            console.log('🔧 Debug: Day result:', dayResult);
            
            if (!dayResult.success) {
                alert(dayResult.error);
                return;
            }

            this.currentPhase = 'results';
            this.showResultsPhase(dayResult);
            this.updateDisplay();
            
            console.log('🔧 Debug: openForBusiness() completed successfully');
            
        } catch (error) {
            console.error('❌ Error in openForBusiness():', error);
            console.error('Stack trace:', error.stack);
            alert(`An error occurred: ${error.message}. Please check the console for details.`);
        }
    }

    /**
     * Show day results
     */
    showResultsPhase(dayResult) {
        const gamePhase = document.getElementById('game-phase');
        const template = document.getElementById('results-phase-template');
        gamePhase.innerHTML = template.innerHTML;

        this.displayDayResults(dayResult);
        this.bindResultsEventHandlers();
    }

    /**
     * Display day results in a grid
     */
    displayDayResults(dayResult) {
        const resultsDiv = document.getElementById('day-results');
        
        const profitClass = dayResult.profit >= 0 ? 'positive' : 'negative';
        
        resultsDiv.innerHTML = `
            <div class="result-card">
                <div class="result-label">Customers Served</div>
                <div class="result-value">${dayResult.customersServed}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Customers Lost</div>
                <div class="result-value">${dayResult.customersLost}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Revenue</div>
                <div class="result-value">$${dayResult.revenue.toFixed(2)}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Operating Cost</div>
                <div class="result-value">$${dayResult.operatingCost.toFixed(2)}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Profit</div>
                <div class="result-value ${profitClass}">$${dayResult.profit.toFixed(2)}</div>
            </div>
            <div class="result-card">
                <div class="result-label">Ending Cash</div>
                <div class="result-value">$${dayResult.endingCash.toFixed(2)}</div>
            </div>
        `;
    }

    /**
     * Bind event handlers for results phase
     */
    bindResultsEventHandlers() {
        document.getElementById('continue-day-btn').addEventListener('click', () => {
            this.continueToNextDay();
        });

        document.getElementById('view-details-btn').addEventListener('click', () => {
            this.showHourlyDetails();
        });
    }

    /**
     * Continue to next day or end game
     */
    continueToNextDay() {
        if (this.game.isGameOver()) {
            this.showGameOver();
        } else {
            this.currentPhase = 'pre-game';
            this.updateDisplay();
            
            // Auto-start next day
            setTimeout(() => {
                this.startNewDay();
            }, 500);
        }
    }

    /**
     * Show game over screen
     */
    showGameOver() {
        const gameOverDiv = document.getElementById('game-over');
        const finalResults = this.game.getFinalResults();
        
        const resultsHTML = `
            <div class="results-grid">
                <div class="result-card">
                    <div class="result-label">Days Played</div>
                    <div class="result-value">${finalResults.daysPlayed}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Final Cash</div>
                    <div class="result-value">$${finalResults.finalCash.toFixed(2)}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Total Profit</div>
                    <div class="result-value ${finalResults.totalProfit >= 0 ? 'positive' : 'negative'}">
                        $${finalResults.totalProfit.toFixed(2)}
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-label">Total Customers</div>
                    <div class="result-value">${finalResults.totalCustomers}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Average Price</div>
                    <div class="result-value">$${finalResults.averagePrice.toFixed(2)}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Inventory Value</div>
                    <div class="result-value">$${finalResults.inventoryValue.toFixed(2)}</div>
                </div>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                ${finalResults.isBankrupt ? 
                    '<p style="color: #e74c3c; font-size: 1.2rem;">💸 You went bankrupt! Better luck next time.</p>' :
                    '<p style="color: #27ae60; font-size: 1.2rem;">🎉 Congratulations! You completed all 30 days!</p>'
                }
                <p style="margin-top: 10px;">Can you beat the AI models? The current leaderboard shows profits around $500-1500.</p>
            </div>
        `;
        
        document.getElementById('final-results').innerHTML = resultsHTML;
        gameOverDiv.style.display = 'block';
        
        // Hide other elements
        document.getElementById('game-phase').style.display = 'none';
        document.getElementById('action-panel').style.display = 'none';
    }

    /**
     * Restart the game
     */
    restartGame() {
        this.initializeGame();
        document.getElementById('game-over').style.display = 'none';
        document.getElementById('game-phase').style.display = 'block';
        document.getElementById('action-panel').style.display = 'block';
    }

    /**
     * Update all display elements
     */
    updateDisplay() {
        // Update game stats
        document.getElementById('current-day').textContent = this.game.currentDay;
        document.getElementById('current-cash').textContent = `$${this.game.cash.toFixed(2)}`;
        
        const totalProfit = this.game.cash - this.game.startingCash;
        document.getElementById('total-profit').textContent = `$${totalProfit.toFixed(2)}`;

        // Update inventory display
        const summary = this.game.inventory.getSummary();
        document.getElementById('cups-count').textContent = summary.cups;
        document.getElementById('lemons-count').textContent = summary.lemons;
        document.getElementById('sugar-count').textContent = summary.sugar;
        document.getElementById('water-count').textContent = summary.water;
        document.getElementById('max-lemonade').textContent = summary.canMake;

        // Update button states
        const startBtn = document.getElementById('start-day-btn');
        const nextBtn = document.getElementById('next-phase-btn');

        if (this.currentPhase === 'pre-game') {
            startBtn.style.display = 'inline-block';
            nextBtn.style.display = 'none';
            startBtn.textContent = this.game.currentDay === 0 ? 'Start New Game' : 'Start Day ' + (this.game.currentDay + 1);
        } else {
            startBtn.style.display = 'none';
            nextBtn.style.display = 'none'; // Hide next button, phase-specific buttons handle flow
        }

        // Show game prompt if pre-game
        if (this.currentPhase === 'pre-game') {
            document.getElementById('game-phase').innerHTML = `
                <h2>🍋 Ready to Start?</h2>
                <div class="phase-content">
                    <p style="font-size: 1.1rem; line-height: 1.6;">${this.game.getTurnPrompt()}</p>
                </div>
            `;
        }
    }

    /**
     * Show expired items alert
     */
    showExpiredItemsAlert(expiredItems) {
        const items = Object.entries(expiredItems).map(([item, count]) => 
            `${count} ${item}`
        ).join(', ');
        
        alert(`⚠️ Expired Inventory: ${items} have expired and been discarded!`);
    }

    /**
     * Show hourly details (could be expanded with charts)
     */
    showHourlyDetails() {
        const lastDay = this.game.history[this.game.history.length - 1];
        if (!lastDay) return;

        let details = "Hourly Breakdown:\n\n";
        for (const [hour, data] of Object.entries(lastDay.hourlySales)) {
            const timeStr = `${hour}:00-${parseInt(hour) + 1}:00`;
            details += `${timeStr}: ${data.served}/${data.demand} customers (${data.lost} lost)\n`;
        }

        alert(details);
    }
}

// Initialize the game when page loads
document.addEventListener('DOMContentLoaded', () => {
    new GameUI();
});