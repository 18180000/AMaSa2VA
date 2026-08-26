#!/usr/bin/env python3
"""
Unit tests for question-conditioned frame-level memory retrieval
"""

import os
import sys
import types
import torch
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock

# Unit tests exercise the wrapper's retrieval code without loading a checkpoint.
# Provide only the upstream base-class contract needed by Sa2VAChatMem.__init__.
vlmeval = types.ModuleType("vlmeval")
vlmeval_vlm = types.ModuleType("vlmeval.vlm")
vlmeval_sa2va = types.ModuleType("vlmeval.vlm.sa2va_chat")


class _FakeSa2VAChat:
    def __init__(self, *args, **kwargs):
        self.pl_enabled = False
        self.pl_adapter = None


vlmeval_sa2va.Sa2VAChat = _FakeSa2VAChat
sys.modules["vlmeval"] = vlmeval
sys.modules["vlmeval.vlm"] = vlmeval_vlm
sys.modules["vlmeval.vlm.sa2va_chat"] = vlmeval_sa2va

from projects.amasa2va.models.internvl.video_qa import Sa2VAChatMem


def test_question_encoding():
    """Test question text encoding into LLM space"""
    print("\n[TEST 1] Question Encoding")

    # Mock components
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_embeddings = Mock()

    # Setup mock tokenizer
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.eos_token_id = 0

    # Setup mock embeddings
    embedding_dim = 768
    mock_emb_tensor = torch.randn(5, embedding_dim)
    mock_embeddings.return_value = mock_emb_tensor
    mock_model.language_model.get_input_embeddings.return_value = mock_embeddings

    # Create instance
    chat = Sa2VAChatMem(model_path="dummy")
    chat.model = mock_model
    chat.tokenizer = mock_tokenizer
    chat.device = torch.device("cpu")

    # Test encoding
    question = "What color is the car?"
    q_repr = chat._encode_question(question)

    assert q_repr.shape == (embedding_dim,), f"Expected shape ({embedding_dim},), got {q_repr.shape}"
    assert torch.allclose(torch.norm(q_repr), torch.tensor(1.0), atol=1e-5), "Question repr should be normalized"

    print("✓ Question encoded successfully")
    print(f"  Shape: {q_repr.shape}")
    print(f"  Norm: {torch.norm(q_repr).item():.4f}")


def test_question_extraction():
    """Test question text extraction from message"""
    print("\n[TEST 2] Question Text Extraction")

    chat = Sa2VAChatMem(model_path="dummy")

    # Test case 1: Question with options (Video-MME style)
    message = [
        {"type": "image", "value": "frame1.jpg"},
        {"type": "text", "value": "Frame-1 Frame-2 Frame-3 What is the person doing?\nA. Walking\nB. Running\nC. Sitting\nD. Standing\nAnswer with the option's letter from the given choices directly."}
    ]

    chat.qa_query_mode = "question_with_options"
    extracted = chat._extract_question_text(message, "Video-MME")

    assert "Frame-" not in extracted, "Should remove Frame-i placeholders"
    assert "Answer with the option's letter" not in extracted, "Should remove instruction suffix"
    print(f"✓ Extracted (with options): {extracted[:80]}...")

    # Test case 2: Question only mode
    chat.qa_query_mode = "question_only"
    extracted = chat._extract_question_text(message, "Video-MME")

    assert "A." not in extracted and "B." not in extracted, "Should remove options in question_only mode"
    print(f"✓ Extracted (question only): {extracted[:80]}...")


def test_frame_retrieval():
    """Test question-conditioned frame retrieval"""
    print("\n[TEST 3] Question-Conditioned Frame Retrieval")

    chat = Sa2VAChatMem(model_path="dummy")
    chat.qa_top_k = 3
    chat.qa_retrieval_temperature = 0.4
    chat.qa_restore_temporal_order = True

    # Mock question encoding
    chat._encode_question = lambda q: F.normalize(torch.randn(768), dim=-1)

    # Create fake frame features: T=8 frames, N=256 tokens, D=768
    T, N, D = 8, 256, 768
    frame_features = torch.randn(T, N, D)

    # Test retrieval
    question = "What happens at the end?"
    idx_sorted, scores, all_sims = chat._retrieve_frames(
        question, frame_features, topk=chat.qa_top_k, tau=chat.qa_retrieval_temperature
    )

    assert len(idx_sorted) == 3, f"Expected 3 frames, got {len(idx_sorted)}"
    assert torch.all(idx_sorted[:-1] <= idx_sorted[1:]), "Frames should be in temporal order"

    print(f"✓ Retrieved frames: {idx_sorted.tolist()}")
    print(f"  Scores: {[f'{s:.3f}' for s in scores.tolist()]}")
    print(f"  All similarities shape: {all_sims.shape}")


def test_topk_select_fusion():
    """Test topk_select fusion mode"""
    print("\n[TEST 4] topk_select Fusion Mode")

    chat = Sa2VAChatMem(model_path="dummy")
    chat.qa_fusion_mode = "topk_select"
    chat.qa_top_k = 4
    chat.qa_retrieval_temperature = 0.4
    chat.device = torch.device("cpu")

    # Mock model and methods
    chat.model = Mock()
    T, N, D = 8, 256, 768
    frame_features = torch.randn(T, N, D)
    chat.model.extract_feature.return_value = frame_features
    chat.model.vision_model.dtype = torch.float32

    chat._encode_question = lambda q: F.normalize(torch.randn(D), dim=-1)

    # Test
    pixel_values = torch.randn(T, 3, 224, 224)
    question = "How many people are there?"

    result, keep_idx = chat._mix_video_features_qa(pixel_values, question)

    assert result.shape[0] == chat.qa_top_k, f"Expected {chat.qa_top_k} frames, got {result.shape[0]}"
    assert result.shape[1:] == (N, D), f"Token shape should be preserved"
    assert keep_idx is not None and len(keep_idx) == chat.qa_top_k

    print(f"✓ topk_select: {T} frames → {result.shape[0]} frames")


def test_weighted_enhance_fusion():
    """Test weighted_enhance fusion mode"""
    print("\n[TEST 5] weighted_enhance Fusion Mode")

    chat = Sa2VAChatMem(model_path="dummy")
    chat.qa_fusion_mode = "weighted_enhance"
    chat.qa_top_k = 3
    chat.qa_retrieval_temperature = 0.4
    chat.vqa_mem_alpha = 0.5
    chat.device = torch.device("cpu")

    # Mock
    chat.model = Mock()
    T, N, D = 6, 256, 768
    frame_features = torch.randn(T, N, D)
    chat.model.extract_feature.return_value = frame_features
    chat.model.vision_model.dtype = torch.float32

    chat._encode_question = lambda q: F.normalize(torch.randn(D), dim=-1)

    # Test
    pixel_values = torch.randn(T, 3, 224, 224)
    question = "What color is the object?"

    result, keep_idx = chat._mix_video_features_qa(pixel_values, question)

    assert result.shape == (T, N, D), f"Should keep all frames, got {result.shape}"
    assert keep_idx is None

    print(f"✓ weighted_enhance: {T} frames → {result.shape[0]} frames (all preserved)")


def test_legacy_compatibility():
    """Test backward compatibility with VQA_MEM_ENABLE"""
    print("\n[TEST 6] Legacy VQA Memory Compatibility")

    os.environ["VQA_MEM_ENABLE"] = "1"
    os.environ["VQA_MEM_TOPK"] = "5"
    os.environ["VQA_MEM_TAU"] = "0.3"
    os.environ["VQA_MEM_ALPHA"] = "0.6"

    chat = Sa2VAChatMem(model_path="dummy")

    assert chat.vqa_mem_enable == True
    assert chat.vqa_mem_topk == 5
    assert chat.vqa_mem_tau == 0.3
    assert chat.vqa_mem_alpha == 0.6

    # Mock
    chat.model = Mock()
    T, N, D = 5, 256, 768
    frame_features = torch.randn(T, N, D)
    chat.model.extract_feature.return_value = frame_features
    chat.model.vision_model.dtype = torch.float32
    chat.device = torch.device("cpu")

    pixel_values = torch.randn(T, 3, 224, 224)
    result = chat._mix_video_features_legacy(pixel_values)

    assert result.shape == (T, N, D), f"Legacy mode should preserve shape"

    print("✓ Legacy mode works correctly")

    # Cleanup
    del os.environ["VQA_MEM_ENABLE"]
    del os.environ["VQA_MEM_TOPK"]
    del os.environ["VQA_MEM_TAU"]
    del os.environ["VQA_MEM_ALPHA"]


def test_config_routing():
    """Test routing between QA and VQA memory based on config"""
    print("\n[TEST 7] Config-Based Routing")

    # Test QA memory enabled
    os.environ["QA_MEM_ENABLE"] = "1"
    os.environ["VQA_MEM_ENABLE"] = "0"

    chat = Sa2VAChatMem(model_path="dummy")
    assert chat.qa_memory_enabled == True
    assert chat.vqa_mem_enable == False
    print("✓ QA memory routing configured")

    del os.environ["QA_MEM_ENABLE"]

    # Test VQA memory enabled
    os.environ["VQA_MEM_ENABLE"] = "1"
    os.environ["QA_MEM_ENABLE"] = "0"

    chat = Sa2VAChatMem(model_path="dummy")
    assert chat.vqa_mem_enable == True
    assert chat.qa_memory_enabled == False
    print("✓ VQA memory routing configured")

    # Cleanup
    del os.environ["VQA_MEM_ENABLE"]
    del os.environ["QA_MEM_ENABLE"]


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("Running QA Memory Unit Tests")
    print("=" * 60)

    try:
        test_question_encoding()
        test_question_extraction()
        test_frame_retrieval()
        test_topk_select_fusion()
        test_weighted_enhance_fusion()
        test_legacy_compatibility()
        test_config_routing()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
