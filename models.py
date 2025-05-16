import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

import math

class MarketModels:
    def __init__(self, buffer_size=100):
        # Initialize data buffers for models
        self.buffer_size = buffer_size
        self.price_history = deque(maxlen=buffer_size)
        self.spread_history = deque(maxlen=buffer_size)
        self.volume_history = deque(maxlen=buffer_size)
        self.volatility = 0.0
        
    def update_market_data(self, price, spread, volume):
        """Update market data buffers with new tick data"""
        self.price_history.append(price)
        self.spread_history.append(spread)
        self.volume_history.append(volume)
        
        # Update volatility estimate if we have enough data
        if len(self.price_history) > 10:
            # Calculate rolling volatility (standard deviation of returns)
            prices = np.array(list(self.price_history))
            returns = np.diff(prices) / prices[:-1]
            self.volatility = np.std(returns) * 100  # Convert to percentage
    
    def estimate_slippage(self, quantity, side, orderbook):
        """
        Estimate slippage using linear regression model based on:
        - Order size
        - Market volatility
        - Current spread
        - Order book depth
        """
        if len(self.price_history) < 20:
            # Not enough data for regression, use basic estimate
            return self._estimate_basic_slippage(quantity, side, orderbook)
        
        try:
            # Create features for regression
            X = np.array([
                [q, self.volatility, s] 
                for q, s in zip(
                    np.linspace(quantity * 0.1, quantity, 10),  # Different order sizes
                    list(self.spread_history)[-10:]  # Recent spreads
                )
            ])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Target variable - simulate slippage values based on order size and spread
            quantity_in_btc = 1000 / float(orderbook['asks'][0][0])  # $1000 worth of BTC

            y = np.array([
                0.0005 * q + 0.05 * s + 0.02 * self.volatility
                for q, s in zip(
                    np.linspace(quantity_in_btc * 0.1, quantity, 10),
                    list(self.spread_history)[-10:]
                )
            ])


            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
           # Predict slippage for current order
            current_spread = self.spread_history[-1] if self.spread_history else orderbook["asks"][0][0] - orderbook["bids"][0][0]
            input_scaled = scaler.transform([[quantity, self.volatility, current_spread]])
            prediction = model.predict(input_scaled)

            # Adjust prediction and cap it
            predicted_slippage = max(0, prediction[0])  # Cap and sanitize

            return predicted_slippage/10000

                
        except Exception as e:
            print(f"Error in slippage regression model: {e}")
            return self._estimate_basic_slippage(quantity, side, orderbook)
    
    def _estimate_basic_slippage(self, quantity, side, orderbook):
        """Fallback method for basic slippage estimation when regression isn't possible"""
        levels = orderbook["asks"] if side.lower() == "buy" else orderbook["bids"]

        if not levels:
            return 0.0

        remaining_qty = quantity
        total_cost = 0.0
        filled_qty = 0.0

        for price_str, size_str in levels:
            price = float(price_str)
            size = float(size_str)

            if remaining_qty <= size:
                total_cost += remaining_qty * price
                filled_qty += remaining_qty
                break
            else:
                total_cost += size * price
                filled_qty += size
                remaining_qty -= size

        if filled_qty == 0:
            return 0.0

        avg_price = total_cost / filled_qty
        top_price = float(levels[0][0])

        if side.lower() == "buy":
            slippage_pct = ((avg_price / top_price) - 1) * 100
        else:
            slippage_pct = ((top_price / avg_price) - 1) * 100

        # Cap unrealistic slippage
        return min(max(0.0, slippage_pct), 5.0)  # Max 5%

    
    def calculate_market_impact(self, quantity, side, price, volatility_override=None):
        """
        Calculate market impact using simplified Almgren-Chriss model
        
        Market impact = permanent_impact + temporary_impact
        
        where:
        - permanent_impact is the lasting effect on the market price
        - temporary_impact is the transient effect during execution
        """
        volatility = volatility_override if volatility_override is not None else self.volatility
      
        sigma = volatility / 100  # Convert percentage to decimal
        market_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else 10000
        
        # Convert quantity to asset units
        quantity_in_asset = quantity / price
        
        # Calculate market participation rate (simplified)
        participation_rate = quantity_in_asset / market_volume
        
        # Parameters for Almgren-Chriss model
        # These are typically calibrated from market data
        gamma = 0.314  # Permanent impact parameter
        eta = 0.142    # Temporary impact parameter
        
        # Calculate permanent impact component
        # This is the lasting price change after the trade
        permanent_impact = gamma * sigma * (quantity_in_asset / market_volume) ** 0.5
        
        # Calculate temporary impact component
        # This is the transient price change during execution
        temporary_impact = eta * sigma * (quantity_in_asset / market_volume) ** 0.6
        
        # Total market impact in price currency units
        total_impact = price * (permanent_impact + temporary_impact)
        
        # Adjust sign based on order side
        if side.lower() == "sell":
            total_impact = -total_impact
            
        return abs(total_impact)
    
    def predict_maker_taker_ratio(self, order_type, quantity, volatility_override=None):
        """
        Predict maker/taker ratio using logistic regression
        Returns probability of being a maker order (0-1)
        """
        # For market orders, always taker
        if order_type.lower() == "market":
            return 0.0  # 0% maker
            
        # For limit orders, use a model
        # Use provided volatility or calculated volatility
        volatility = volatility_override if volatility_override is not None else self.volatility
        
        try:
            # If we have enough data, build a logistic model
            if len(self.price_history) > 5:
                # Create synthetic training data for demonstration
                # In a real system, this would use historical order execution data
                X_train = np.array([
                    [vol, q] for vol, q in zip(
                        np.random.uniform(0, volatility * 2, 100),  # Volatility values
                        np.random.uniform(0, quantity * 2, 100)     # Quantity values
                    )
                ])
                
                # Generate target classes: 1 for maker, 0 for taker
                # Higher volatility and quantity decrease maker probability
                y_train = np.array([
                    1 if vol < volatility * 0.8 and q < quantity * 0.7 else 0
                    for vol, q in X_train
                ])
                
                # Fit logistic regression model
                model = LogisticRegression()
                model.fit(X_train, y_train)
                
                # Predict maker probability for current order
                maker_prob = model.predict_proba([[volatility, quantity]])[0][1]
                return maker_prob
            else:
                # Not enough data, use heuristic
                # Higher volatility = lower maker probability
                # Higher quantity = lower maker probability
                base_prob = 0.7  # Base probability for limit orders
                vol_factor = min(1, volatility / 5)  # Reduce probability as volatility increases
                qty_factor = min(1, quantity / 1000)  # Reduce probability as quantity increases
                
                maker_prob = base_prob * (1 - vol_factor * 0.5) * (1 - qty_factor * 0.5)
                return maker_prob
                
        except Exception as e:
            print(f"Error in maker/taker model: {e}")
            # Fallback to simple heuristic
            return 0.5  # 50/50 for limit orders as fallback
    
    def calculate_fees(self, quantity, fee_tier, maker_ratio=0.0):
        """
        Calculate fees based on OKX fee structure and maker/taker ratio
        
        Fee tiers (taker/maker):
        - Tier 1: 0.08% / 0.05%
        - Tier 2: 0.06% / 0.03%
        - Tier 3: 0.04% / 0.01%
        """
        # Define fee rates based on tier
        if "1" in fee_tier:
            taker_fee = 0.0008
            maker_fee = 0.0005
        elif "2" in fee_tier:
            taker_fee = 0.0006
            maker_fee = 0.0003
        elif "3" in fee_tier:
            taker_fee = 0.0004
            maker_fee = 0.0001
        else:
            # Default to highest fees
            taker_fee = 0.0008
            maker_fee = 0.0005
            
        # Calculate blended fee rate
        fee_rate = (maker_ratio * maker_fee) + ((1 - maker_ratio) * taker_fee)
        
        # Calculate fee amount
        fee_amount = quantity * fee_rate
        
        return fee_amount, fee_rate
        
    def calculate_net_cost(self, quantity, slippage_pct, market_impact, fee_amount, side):
        """
        Calculate net cost of trade
        
        For buy:  net_cost = quantity * (1 + slippage_pct/100) + market_impact + fee_amount
        For sell: net_proceeds = quantity * (1 - slippage_pct/100) - market_impact - fee_amount
        """
        slippage_amount = quantity * (slippage_pct / 100)
        
        if side.lower() == "buy":
            return quantity + slippage_amount + market_impact + fee_amount
        else:
            return quantity - slippage_amount - market_impact - fee_amount
        
    def update(self, price=None, spread=None, volume=None, reset=False):
        
        if reset:
            self.price_history.clear()
            self.spread_history.clear()
            self.volume_history.clear()
            self.volatility = 0.0
            return

        if price is not None:
            self.price_history.append(price)
        if spread is not None:
            self.spread_history.append(spread)
        if volume is not None:
            self.volume_history.append(volume)

        # Recalculate volatility if we have enough data
        if len(self.price_history) > 10:
            prices = np.array(list(self.price_history))
            returns = np.diff(prices) / prices[:-1]
            self.volatility = np.std(returns) * 100  # percent
