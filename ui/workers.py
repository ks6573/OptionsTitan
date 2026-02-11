"""
OptionsTitan QThread Workers
Background threads for long-running operations
"""

from PySide6.QtCore import QThread, Signal, QObject
import traceback


class TickerValidationWorker(QThread):
    """
    Validates a stock symbol via yfinance in a background thread.
    Does not block the UI.
    """

    validation_passed = Signal(str)
    validation_failed = Signal(str, str)  # symbol, error_message

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = (symbol or "").strip().upper()

    def run(self):
        if not self.symbol:
            self.validation_failed.emit(self.symbol, "Symbol cannot be empty.")
            return
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="5d")
            if hist is None or hist.empty:
                self.validation_failed.emit(
                    self.symbol,
                    f"No market data found for '{self.symbol}'. The symbol may be invalid or delisted.",
                )
                return
            self.validation_passed.emit(self.symbol)
        except Exception as e:
            self.validation_failed.emit(
                self.symbol,
                f"Validation failed: {str(e)}",
            )


class CoverageCheckWorker(QThread):
    """
    Checks dataset coverage for a symbol in background.
    Emits coverage_result(str, bool) when done.
    """

    coverage_result = Signal(str, bool)  # symbol, has_coverage

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = (symbol or "").strip()

    def run(self):
        try:
            from src.data_collection.dataset_catalog import has_dataset_coverage
            has_coverage = has_dataset_coverage(self.symbol)
            self.coverage_result.emit(self.symbol, has_coverage)
        except Exception:
            self.coverage_result.emit(self.symbol, False)


class CatalogLoaderWorker(QThread):
    """
    Loads available tickers from dataset catalog in background.
    Used to populate autocomplete.
    """

    catalog_loaded = Signal(object)  # set of ticker strings

    def run(self):
        try:
            from src.data_collection.dataset_catalog import get_available_dataset_tickers
            tickers = get_available_dataset_tickers()
            self.catalog_loaded.emit(tickers)
        except Exception:
            self.catalog_loaded.emit(set())


class AnalysisWorker(QThread):
    """
    Worker thread for running options strategy analysis in the background.
    This prevents the GUI from freezing during analysis.
    """
    
    # Signals for communication with main thread
    progress_update = Signal(str)  # Progress message
    analysis_complete = Signal(dict, list)  # stock_data, strategies
    analysis_error = Signal(str)  # Error message
    
    def __init__(self, analyzer, symbol, liquidity, max_risk, target_profit, max_loss):
        """
        Initialize the analysis worker.
        
        Args:
            analyzer: OptionsStrategyAnalyzer instance
            symbol: Stock symbol
            liquidity: Portfolio liquidity
            max_risk: Maximum risk percentage
            target_profit: Target profit percentage
            max_loss: Maximum loss percentage
        """
        super().__init__()
        self.analyzer = analyzer
        self.symbol = symbol
        self.liquidity = liquidity
        self.max_risk = max_risk
        self.target_profit = target_profit
        self.max_loss = max_loss
    
    def run(self):
        """Execute the analysis in the background thread"""
        try:
            # Step 1: Fetch stock data
            self.progress_update.emit(f"Fetching market data for {self.symbol}...")
            stock_data = self.analyzer.get_stock_data(self.symbol)
            
            # Step 2: Generate strategies
            self.progress_update.emit("Analyzing strategies...")
            strategies = self.analyzer.generate_strategies(
                stock_data,
                self.liquidity,
                self.max_risk,
                self.target_profit,
                self.max_loss
            )
            
            # Step 3: Complete
            self.progress_update.emit("Analysis complete!")
            self.analysis_complete.emit(stock_data, strategies)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Log to console
            self.analysis_error.emit(str(e))


class LLAMAInsightWorker(QThread):
    """
    Worker thread for generating LLAMA AI insights.
    Runs separately to provide progressive feedback.
    """
    
    # Signals
    insight_ready = Signal(str, str)  # insight_type, insight_text
    insight_error = Signal(str)  # Error message
    
    def __init__(self, llama_enhancer, insight_type, **kwargs):
        """
        Initialize LLAMA insight worker.
        
        Args:
            llama_enhancer: LLAMAEnhancer instance
            insight_type: Type of insight ('market', 'strategy', 'risk', 'comparison')
            **kwargs: Additional arguments for the specific insight type
        """
        super().__init__()
        self.llama_enhancer = llama_enhancer
        self.insight_type = insight_type
        self.kwargs = kwargs
    
    def run(self):
        """Generate the requested insight"""
        try:
            if not self.llama_enhancer.enabled:
                self.insight_error.emit("LLAMA AI not enabled")
                return
            
            insight = None
            
            if self.insight_type == 'market':
                insight = self.llama_enhancer.get_market_insights(
                    self.kwargs['stock_data'],
                    self.kwargs['user_params']
                )
            elif self.insight_type == 'strategy':
                insight = self.llama_enhancer.enhance_strategy_reasoning(
                    self.kwargs['strategy'],
                    self.kwargs['stock_data'],
                    self.kwargs['user_params']
                )
            elif self.insight_type == 'risk':
                insight = self.llama_enhancer.get_risk_assessment_commentary(
                    self.kwargs['strategies'],
                    self.kwargs['user_params']
                )
            elif self.insight_type == 'comparison':
                insight = self.llama_enhancer.compare_strategies(
                    self.kwargs['strategy1'],
                    self.kwargs['strategy2'],
                    self.kwargs['stock_data']
                )
            
            if insight:
                self.insight_ready.emit(self.insight_type, insight)
            else:
                self.insight_error.emit(f"No insight generated for {self.insight_type}")
                
        except Exception as e:
            error_msg = f"LLAMA insight failed: {str(e)}"
            print(error_msg)
            self.insight_error.emit(str(e))
