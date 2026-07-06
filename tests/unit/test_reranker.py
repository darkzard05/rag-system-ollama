import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.core.reranker import (
    DistributedReranker,
    RerankerStrategy,
    RerankingMetrics,
    FlashReranker,
    RerankingResult
)

class TestDistributedReranker:

    @pytest.fixture
    def reranker(self):
        """Provides a DistributedReranker instance with mocked config."""
        with patch("src.core.reranker.RERANKER_ENABLED", True), \
             patch("src.core.reranker.RERANKER_CONFIG", {"bypass_threshold": 0.95, "top_k": 10}):
            # Mock FlashReranker to avoid actual initialization
            with patch("src.core.reranker.FlashReranker") as mock_flash:
                instance = DistributedReranker()
                instance.flash_engine = mock_flash.return_value
                return instance

    def test_rerank_empty_results(self, reranker):
        """Test case: Empty results list."""
        results, metrics = reranker.rerank([], query_text="test query")
        
        assert results == []
        assert metrics.total_results == 0
        assert metrics.strategy_used == RerankerStrategy.SEMANTIC_FLASH.value

    def test_rerank_single_result(self, reranker):
        """Test case: Single result (should return as is)."""
        doc = Document(page_content="test content", metadata={"score": 0.9})
        # We need to attach score to the object as the code uses getattr(results[0], "score", 0.0)
        # For Document, it's usually in metadata, but the code looks for an attribute.
        # Let's use a Mock or a custom class to simulate the attribute.
        mock_doc = MagicMock(spec=Document)
        mock_doc.page_content = "test content"
        mock_doc.metadata = {"score": 0.9}
        mock_doc.score = 0.9
        
        results, metrics = reranker.rerank([mock_doc], query_text="test query")
        
        assert len(results) == 1
        assert results[0] == mock_doc
        assert metrics.total_results == 1

    def test_rerank_early_exit_trigger(self, reranker):
        """Test case: Early Exit trigger (First score 0.96, second 0.5 -> Gap 0.46 -> skip rerank)."""
        doc1 = MagicMock(spec=Document)
        doc1.score = 0.96
        doc2 = MagicMock(spec=Document)
        doc2.score = 0.5
        
        results, metrics = reranker.rerank([doc1, doc2], query_text="test query")
        
        assert metrics.strategy_used == "early_exit_bypass"
        assert results == [doc1, doc2]
        reranker.flash_engine.rerank_documents.assert_not_called()

    def test_rerank_early_exit_failure(self, reranker):
        """Test case: Early Exit failure (First score 0.96, second 0.90 -> Gap 0.06 -> perform rerank)."""
        doc1 = MagicMock(spec=Document)
        doc1.score = 0.96
        doc2 = MagicMock(spec=Document)
        doc2.score = 0.90
        
        # Mock rerank_documents to return the same docs
        reranker.flash_engine.rerank_documents.return_value = [doc1, doc2]
        
        results, metrics = reranker.rerank([doc1, doc2], query_text="test query")
        
        assert metrics.strategy_used == RerankerStrategy.SEMANTIC_FLASH.value
        assert len(results) == 2
        reranker.flash_engine.rerank_documents.assert_called_once()

    def test_rerank_semantic_flash_flow(self, reranker):
        """Test case: Semantic Flash flow (verify `rerank_documents` is called and results are returned)."""
        doc1 = Document(page_content="doc1", metadata={"score": 0.8})
        doc2 = Document(page_content="doc2", metadata={"score": 0.7})
        
        reranked_doc1 = Document(page_content="doc1_reranked", metadata={"score": 0.9})
        reranked_doc2 = Document(page_content="doc2_reranked", metadata={"score": 0.8})
        
        reranker.flash_engine.rerank_documents.return_value = [reranked_doc1, reranked_doc2]
        
        results, metrics = reranker.rerank([doc1, doc2], query_text="test query")
        
        assert len(results) == 2
        assert results[0].page_content == "doc1_reranked"
        assert metrics.reranked_results == 2
        reranker.flash_engine.rerank_documents.assert_called_once()

    def test_rerank_fallback_score_only(self, reranker):
        """Test case: Fallback to score-based sorting when reranker is disabled or strategy is SCORE_ONLY."""
        doc1 = MagicMock(spec=Document)
        doc1.score = 0.7
        doc1.page_content = "doc1"
        doc1.metadata = {}
        
        doc2 = MagicMock(spec=Document)
        doc2.score = 0.9
        doc2.page_content = "doc2"
        doc2.metadata = {}
        
        # Use SCORE_ONLY strategy
        results, metrics = reranker.rerank([doc1, doc2], query_text="test query", strategy=RerankerStrategy.SCORE_ONLY)
        
        assert results[0].score == 0.9
        assert results[1].score == 0.7
        assert metrics.strategy_used == RerankerStrategy.SCORE_ONLY.value
        reranker.flash_engine.rerank_documents.assert_not_called()

    def test_rerank_input_type_flexibility(self, reranker):
        """Test case: Input type flexibility (mix of `Document` and other objects)."""
        doc = Document(page_content="doc", metadata={"score": 0.8})
        
        # RerankingResult is a dataclass, we can mock it or use it
        res_obj = RerankingResult(
            doc_id="1",
            content="res_content",
            original_score=0.7,
            reranked_score=0.75,
            original_rank=1,
            final_rank=1,
            metadata={}
        )
        
        # The code uses getattr(r, "page_content", getattr(r, "content", ""))
        # For RerankingResult, it has 'content'.
        
        reranker.flash_engine.rerank_documents.return_value = [doc]
        
        results, metrics = reranker.rerank([doc, res_obj], query_text="test query")
        
        assert len(results) == 1 # Because rerank_documents returns top_k
        reranker.flash_engine.rerank_documents.assert_called_once()
        
        # Check if the second object was correctly converted to Document
        # The call to rerank_documents should have received a list of 2 Documents
        args, kwargs = reranker.flash_engine.rerank_documents.call_args
        passed_docs = args[1]
        assert len(passed_docs) == 2
        assert isinstance(passed_docs[0], Document)
        assert passed_docs[0].page_content == "doc"
        assert isinstance(passed_docs[1], Document)
        assert passed_docs[1].page_content == "res_content"

if __name__ == "__main__":
    pytest.main([__file__])
