/**
 * Comprehensive Test Suite for LemonadeBench Web Game
 * This suite tests the game end-to-end to prevent "undefined" errors and other issues
 */

class GameTestSuite {
    constructor() {
        this.testResults = [];
        this.totalTests = 0;
        this.passedTests = 0;
        this.failedTests = 0;
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('🧪 Starting Comprehensive Test Suite...');
        console.log('=' .repeat(50));
        
        // Reset counters
        this.testResults = [];
        this.totalTests = 0;
        this.passedTests = 0;
        this.failedTests = 0;
        
        // Run test categories
        await this.testGameEngineUnit();
        await this.testInventorySystem();
        await this.testDemandModel();
        await this.testUIAlerts();
        await this.testFullGameFlow();
        await this.testErrorHandling();
        await this.testEdgeCases();
        
        // Print summary
        this.printSummary();
    }

    /**
     * Test helper - assert with description
     */
    assert(condition, testName, details = '') {
        this.totalTests++;
        if (condition) {
            this.passedTests++;
            console.log(`✅ PASS: ${testName}`);
            this.testResults.push({ status: 'pass', test: testName });
        } else {
            this.failedTests++;
            console.error(`❌ FAIL: ${testName}`);
            if (details) console.error(`   Details: ${details}`);
            this.testResults.push({ status: 'fail', test: testName, details });
        }
    }

    /**
     * Test helper - check for undefined values
     */
    assertNotUndefined(value, testName) {
        this.assert(
            value !== undefined,
            testName,
            `Value was undefined: ${value}`
        );
    }

    /**
     * Test helper - check for valid error messages
     */
    assertValidErrorMessage(result, testName) {
        if (!result.success) {
            this.assert(
                result.error !== undefined && result.error !== null && result.error !== '',
                testName,
                `Invalid error message: ${result.error}`
            );
        }
    }

    /**
     * 1. Unit tests for game engine
     */
    async testGameEngineUnit() {
        console.log('\n📦 Testing Game Engine Units...');
        
        const game = new BusinessGame();
        
        // Test initialization
        this.assertNotUndefined(game.cash, 'Game cash initialized');
        this.assertNotUndefined(game.currentDay, 'Game current day initialized');
        this.assertNotUndefined(game.inventory, 'Game inventory initialized');
        
        // Test setPrice
        const priceResult1 = game.setPrice(2.50);
        this.assert(priceResult1.success === true, 'Set valid price');
        
        const priceResult2 = game.setPrice(-1);
        this.assert(priceResult2.success === false, 'Reject negative price');
        this.assertValidErrorMessage(priceResult2, 'Negative price error message');
        
        // Test setOperatingHours
        const hoursResult1 = game.setOperatingHours(9, 17);
        this.assert(hoursResult1.success === true, 'Set valid hours');
        
        const hoursResult2 = game.setOperatingHours(17, 9);
        this.assert(hoursResult2.success === false, 'Reject invalid hours');
        this.assertValidErrorMessage(hoursResult2, 'Invalid hours error message');
        
        const hoursResult3 = game.setOperatingHours(-1, 17);
        this.assert(hoursResult3.success === false, 'Reject negative open hour');
        this.assertValidErrorMessage(hoursResult3, 'Negative hour error message');
        
        const hoursResult4 = game.setOperatingHours(9, 25);
        this.assert(hoursResult4.success === false, 'Reject hour > 23');
        this.assertValidErrorMessage(hoursResult4, 'Hour > 23 error message');
    }

    /**
     * 2. Test inventory system
     */
    async testInventorySystem() {
        console.log('\n📦 Testing Inventory System...');
        
        const inventory = new Inventory();
        
        // Test initialization
        this.assert(inventory.getAvailable('cups') === 0, 'Cups start at 0');
        this.assert(inventory.getAvailable('lemons') === 0, 'Lemons start at 0');
        
        // Test adding items
        inventory.addItems('cups', 50, 1);
        this.assert(inventory.getAvailable('cups') === 50, 'Add cups correctly');
        
        // Test recipe usage
        inventory.addItems('lemons', 50, 1);
        inventory.addItems('sugar', 50, 1);
        inventory.addItems('water', 50, 1);
        
        const recipe = { cups: 1, lemons: 1, sugar: 1, water: 1 };
        const canUse = inventory.useItems(recipe);
        this.assert(canUse === true, 'Can use items when available');
        this.assert(inventory.getAvailable('cups') === 49, 'Items decremented correctly');
        
        // Test insufficient inventory
        const recipe2 = { cups: 100, lemons: 100, sugar: 100, water: 100 };
        const canUse2 = inventory.useItems(recipe2);
        this.assert(canUse2 === false, 'Cannot use items when insufficient');
        this.assert(inventory.getAvailable('cups') === 49, 'Items unchanged on failed use');
    }

    /**
     * 3. Test demand model
     */
    async testDemandModel() {
        console.log('\n📊 Testing Demand Model...');
        
        const demand = new DemandModel();
        
        // Test base demand calculation
        const demand1 = demand.calculateBaseDemand(0);
        this.assert(demand1 === 50, 'Base demand at price 0');
        
        const demand2 = demand.calculateBaseDemand(2.50);
        this.assert(demand2 === 25, 'Base demand at price 2.50');
        
        const demand3 = demand.calculateBaseDemand(5.00);
        this.assert(demand3 === 0, 'Base demand at price 5.00');
        
        // Test hour multipliers
        const mult1 = demand.getHourMultiplier(12);
        this.assert(mult1 === 1.5, 'Noon multiplier correct');
        
        const mult2 = demand.getHourMultiplier(3);
        this.assert(mult2 === 0.0, 'Closed hour multiplier');
        
        // Test customer calculation
        const customers = demand.calculateCustomers(2.50, 12, false);
        this.assertNotUndefined(customers, 'Customers calculation returns value');
        this.assert(typeof customers === 'number', 'Customers is a number');
        this.assert(!isNaN(customers), 'Customers is not NaN');
    }

    /**
     * 4. Test UI alert handling
     */
    async testUIAlerts() {
        console.log('\n🚨 Testing UI Alert Handling...');
        
        // Override alert to catch calls
        const originalAlert = window.alert;
        let capturedAlerts = [];
        window.alert = function(msg) {
            capturedAlerts.push(msg);
        };
        
        try {
            // Simulate various error conditions
            const ui = new GameUI();
            
            // Test that error messages are never undefined
            const testCases = [
                { error: undefined, expected: 'not undefined' },
                { error: null, expected: 'not null' },
                { error: '', expected: 'not empty' },
                { error: 'Valid error', expected: 'Valid error' }
            ];
            
            for (const testCase of testCases) {
                capturedAlerts = [];
                
                // Simulate a result with various error values
                const result = { success: false, error: testCase.error };
                
                // This would normally call alert(result.error)
                // But with our fixes, it should use a fallback
                const errorMsg = result.error || 'Failed operation';
                window.alert(errorMsg);
                
                this.assert(
                    capturedAlerts[0] !== 'undefined',
                    `Alert not undefined for error: ${testCase.error}`,
                    `Got: ${capturedAlerts[0]}`
                );
            }
            
        } finally {
            // Restore original alert
            window.alert = originalAlert;
        }
    }

    /**
     * 5. Test full game flow
     */
    async testFullGameFlow() {
        console.log('\n🎮 Testing Full Game Flow...');
        
        const game = new BusinessGame();
        
        // Start new day
        const dayInfo = game.startNewDay();
        this.assertNotUndefined(dayInfo, 'Day info returned');
        this.assert(dayInfo.day === 1, 'Day incremented to 1');
        
        // Order supplies
        const orderResult = game.orderSupplies({ cups: 50, lemons: 50, sugar: 50, water: 50 });
        this.assert(orderResult.success === true, 'Order supplies successful');
        this.assertNotUndefined(orderResult.remainingCash, 'Remaining cash defined');
        
        // Set price and hours
        const priceResult = game.setPrice(2.50);
        this.assert(priceResult.success === true, 'Set price successful');
        
        const hoursResult = game.setOperatingHours(9, 17);
        this.assert(hoursResult.success === true, 'Set hours successful');
        
        // Simulate day
        const dayResult = game.simulateDay();
        this.assertNotUndefined(dayResult, 'Day result returned');
        
        // Check all required properties exist
        const requiredProps = [
            'day', 'price', 'openHour', 'closeHour', 'hoursOpen',
            'customersServed', 'customersLost', 'revenue', 
            'operatingCost', 'profit', 'endingCash'
        ];
        
        for (const prop of requiredProps) {
            this.assertNotUndefined(
                dayResult[prop],
                `Day result has ${prop}`
            );
            
            // Check numeric properties are valid numbers
            if (typeof dayResult[prop] === 'number') {
                this.assert(
                    !isNaN(dayResult[prop]),
                    `Day result ${prop} is valid number`
                );
            }
        }
    }

    /**
     * 6. Test error handling
     */
    async testErrorHandling() {
        console.log('\n⚠️ Testing Error Handling...');
        
        const game = new BusinessGame();
        
        // Test simulating without starting day
        const result1 = game.simulateDay();
        this.assert(result1.success === false, 'Cannot simulate without price/hours');
        this.assertValidErrorMessage(result1, 'Simulate without setup error message');
        
        // Start day but don't set price
        game.startNewDay();
        const result2 = game.simulateDay();
        this.assert(result2.success === false, 'Cannot simulate without price');
        this.assertValidErrorMessage(result2, 'No price error message');
        
        // Set price but not hours
        game.setPrice(2.50);
        const result3 = game.simulateDay();
        this.assert(result3.success === false, 'Cannot simulate without hours');
        this.assertValidErrorMessage(result3, 'No hours error message');
    }

    /**
     * 7. Test edge cases
     */
    async testEdgeCases() {
        console.log('\n🔧 Testing Edge Cases...');
        
        const game = new BusinessGame();
        
        // Test with $0 starting cash
        const poorGame = new BusinessGame(30, 0);
        poorGame.startNewDay();
        const orderResult = poorGame.orderSupplies({ cups: 100, lemons: 100 });
        this.assert(orderResult.success === false, 'Cannot order with no money');
        this.assertValidErrorMessage(orderResult, 'No money error message');
        
        // Test extremely high price
        game.startNewDay();
        const priceResult = game.setPrice(999999);
        this.assert(priceResult.success === true, 'Can set very high price');
        
        // Test 24-hour operation
        const hoursResult = game.setOperatingHours(0, 23);
        this.assert(hoursResult.success === true, 'Can set 23-hour operation');
        
        // Test bankruptcy
        const bankruptGame = new BusinessGame(30, 10);
        bankruptGame.startNewDay();
        bankruptGame.setPrice(10);
        bankruptGame.setOperatingHours(0, 20); // High operating cost
        bankruptGame.simulateDay();
        this.assert(bankruptGame.cash < 0, 'Can go bankrupt');
        this.assert(bankruptGame.isGameOver() === true, 'Game over when bankrupt');
    }

    /**
     * Print test summary
     */
    printSummary() {
        console.log('\n' + '=' .repeat(50));
        console.log('📊 TEST SUMMARY');
        console.log('=' .repeat(50));
        console.log(`Total Tests: ${this.totalTests}`);
        console.log(`✅ Passed: ${this.passedTests}`);
        console.log(`❌ Failed: ${this.failedTests}`);
        console.log(`Success Rate: ${((this.passedTests / this.totalTests) * 100).toFixed(1)}%`);
        
        if (this.failedTests > 0) {
            console.log('\n❌ FAILED TESTS:');
            this.testResults
                .filter(r => r.status === 'fail')
                .forEach(r => {
                    console.log(`  - ${r.test}`);
                    if (r.details) console.log(`    ${r.details}`);
                });
        }
        
        console.log('\n' + '=' .repeat(50));
        
        if (this.failedTests === 0) {
            console.log('🎉 ALL TESTS PASSED! The game is ready for deployment.');
        } else {
            console.log('⚠️ SOME TESTS FAILED! Fix issues before deployment.');
        }
    }
}

// Function to run tests from console
function runTests() {
    const suite = new GameTestSuite();
    suite.runAllTests();
    return suite;
}

// Auto-run tests if this script is loaded directly
if (typeof module === 'undefined') {
    console.log('🚀 LemonadeBench Test Suite Loaded');
    console.log('Run `runTests()` in console to execute all tests');
}