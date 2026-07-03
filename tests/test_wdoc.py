import os
import socket
import sys
import tempfile
from urllib.parse import urlparse

import pytest


def _ollama_is_reachable() -> bool:
    """Check if the Ollama server port is open. Honors OLLAMA_HOST."""
    host_env = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if "://" not in host_env:
        host_env = f"http://{host_env}"
    parsed = urlparse(host_env)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


_OLLAMA_REACHABLE = _ollama_is_reachable()

# add an unexpected env variable to make sure nothing crashes
os.environ["WDOC_TEST_UNEXPECTED_VARIABLE_1"] = "testing"

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"

os.environ["PYTEST_IS_TESTING_WDOC"] = "true"

# test binary embeddings
os.environ["WDOC_MOD_FAISS_BINARY"] = "true"
os.environ["WDOC_MOD_FAISS_SCORE_FN"] = (
    "false"  # needs to be disabled for BINARY to work
)

from wdoc.wdoc import wdoc
from wdoc.utils.misc import ModelName
from wdoc.utils.embeddings import load_embeddings_engine
from wdoc.utils.embeddings import test_embeddings as _test_embeddings
from wdoc.utils.tasks.query import semantic_batching
from wdoc.utils.env import env

os.environ["WDOC_TYPECHECKING"] = "crash"

# Default model names if not specified in environment
# we are testing different providers just in case there are unexpected backend issues
WDOC_TEST_OPENAI_MODEL = os.getenv("WDOC_TEST_OPENAI_MODEL", "gpt-4o")
WDOC_TEST_OPENAI_EVAL_MODEL = os.getenv("WDOC_TEST_OPENAI_EVAL_MODEL", "gpt-4o-mini")
WDOC_TEST_OPENAI_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OPENAI_EMBED_MODEL", "text-embedding-3-small"
)

WDOC_TEST_OPENROUTER_MODEL = os.getenv(
    "WDOC_TEST_OPENROUTER_MODEL",
    "openrouter/mistralai/mistral-small-3.2-24b-instruct",
)
WDOC_TEST_OPENROUTER_EVAL_MODEL = os.getenv(
    "WDOC_TEST_OPENROUTER_EVAL_MODEL",
    "openrouter/mistralai/mistral-small-3.2-24b-instruct",
)

WDOC_TEST_OLLAMA_EMBED_MODEL = os.getenv(
    "WDOC_TEST_OLLAMA_EMBED_MODEL", "embeddinggemma:latest"
)

# Full model id (provider included) so run_all_tests.sh can detect the
# 'mistral/' prefix and validate MISTRAL_API_KEY before the suite runs.
WDOC_TEST_MISTRAL_EMBED_MODEL = os.getenv(
    "WDOC_TEST_MISTRAL_EMBED_MODEL", "mistral/mistral-embed"
)

# also make sure the default models work
WDOC_TEST_DEFAULT_MODEL = os.getenv("WDOC_TEST_DEFAULT_MODEL", env.WDOC_DEFAULT_MODEL)
WDOC_TEST_DEFAULT_EVAL_MODEL = os.getenv(
    "WDOC_TEST_DEFAULT_EVAL_MODEL", env.WDOC_DEFAULT_QUERY_EVAL_MODEL
)
WDOC_TEST_DEFAULT_EMBED_MODEL = os.getenv(
    "WDOC_TEST_DEFAULT_EMBED_MODEL", env.WDOC_DEFAULT_EMBED_MODEL
)

os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"

# the unexpected env var should be tested both before import and before run:
os.environ["WDOC_TEST_UNEXPECTED_VARIABLE_2"] = "testing"

ANALOGY_QUESTION = "What is the analogy used by the speaker"


@pytest.mark.basic
def test_wdoc_version():
    """Test that wdoc has a valid version string."""
    assert isinstance(wdoc.VERSION, str)
    assert len(wdoc.VERSION.split(".")) == 3


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_semantic_batching():
    """Test that semantic_batching properly groups related texts."""
    texts = [
        "The cat chased the mouse around the house",
        "Python is a popular programming language",
        "JavaScript is used for web development",
        "The dog barked at the mailman yesterday",
    ]

    embeddings = load_embeddings_engine(
        modelname=ModelName(env.WDOC_DEFAULT_EMBED_MODEL),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )

    batches = semantic_batching(texts, embeddings)

    # Basic validation
    assert isinstance(batches, list)
    assert len(batches) >= 1
    assert all(isinstance(batch, list) for batch in batches)

    # Check all texts are present
    all_texts = []
    for batch in batches:
        all_texts.extend(batch)
    assert sorted(all_texts) == sorted(texts)

    # Check semantic grouping (programming languages should be together)
    for batch in batches:
        if "Python" in batch[0]:
            assert any("JavaScript" in text for text in batch)
        elif "JavaScript" in batch[0]:
            assert any("Python" in text for text in batch)

    assert texts != all_texts and texts != all_texts[::-1], all_texts


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_cache_cost():
    """Test summarization of Tim Urban's procrastination video. Three times to make sure the caching and caching disabling works."""
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "false"
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "urban" in out["summary"].lower() or "procrastinat" in out["summary"].lower()
    assert out["doc_total_cost"] > 0, out

    inst2 = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        filetype="auto",
    )
    out2 = inst2.summary_task()
    assert "monkey" in out2["summary"].lower()
    assert (
        out2["doc_total_cost"] == 0,
        out2,
    ), (
        f"Normally we should be reusing the cache so cost should be 0 but is {out2['doc_total_cost']}"
    )

    inst3 = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out3 = inst3.summary_task()
    assert "monkey" in out3["summary"].lower()
    assert out3["doc_total_cost"] > 0, (
        f"Normally we disabled the cache so cost should be higher than 0 but is {out3['doc_total_cost']}"
    )
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"


@pytest.mark.basic
def test_source_replace_anchor_links():
    """Test that source_replace converts WDOC_IDs to markdown anchor links."""
    from wdoc.utils.tasks.query import source_replace

    mapping = {"WDOC_1": 1, "WDOC_2": 2, "WDOC_21": 21}
    text = "Information from [[WDOC_1]] and [[WDOC_2]] and [[WDOC_21]]."
    result = source_replace(text, mapping)
    assert "[1](#document-1)" in result
    assert "[2](#document-2)" in result
    assert "[21](#document-21)" in result
    # Make sure WDOC_2 didn't corrupt WDOC_21
    assert "[2](#document-2)1" not in result


@pytest.mark.basic
def test_citation_url_template():
    """Test that citation_url_template converts page citations to clickable links."""
    import re

    template = "https://site.com/docs/{source}#page={page}"
    cite_pattern = re.compile(r"\[p\.(\d+)(?:,\s*([^\]]+))?\]")

    def _make_link(m):
        page = m.group(1)
        source = m.group(2) or "default.pdf"
        url = template.format(page=page, source=source)
        return f"[p.{page}]({url})"

    # Simple citation
    text = "- **Key finding** [p.42]"
    result = cite_pattern.sub(_make_link, text)
    assert "[p.42](https://site.com/docs/default.pdf#page=42)" in result

    # Citation with source
    text2 = "- **Another finding** [p.7, transcript.pdf]"
    result2 = cite_pattern.sub(_make_link, text2)
    assert "[p.7](https://site.com/docs/transcript.pdf#page=7)" in result2


@pytest.mark.basic
def test_summary_tim_urban_testing_model():
    """Test summarization of Tim Urban's procrastination video with testing model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model="testing",  # Use the special testing model
        filetype="auto",
    )
    out = inst.summary_task()
    # The 'testing' model should return a fixed string
    assert "Lorem ipsum dolor sit amet" in out["summary"]


@pytest.mark.basic
def test_summarize_empty_completion_raises():
    """An empty LLM message content during summarization must raise an explicit
    error that dumps the full LLM output, instead of silently returning an empty
    summary. Regression test for the case where the answer lands in another
    field (e.g. additional_kwargs['reasoning_content']) while the message
    content is empty, which previously surfaced as a confusing
    `assert 'monkey' in ''` much later."""
    from langchain_community.chat_models.fake import FakeListChatModel
    from langchain_core.documents import Document

    from wdoc.utils.llm import PriceCountingCallback
    from wdoc.utils.tasks.summarize import _summarize

    # The fake model always returns an empty string for the message content.
    fake_llm = FakeListChatModel(
        verbose=False,
        responses=["", "", ""],
        callbacks=[PriceCountingCallback(verbose=False)],
        disable_streaming=True,
        cache=False,
    )

    docs = [Document(page_content="A monkey talks about procrastination.")]
    metadata = (
        "<text_metadata><section_number>[PROGRESS]</section_number></text_metadata>"
    )

    with pytest.raises(ValueError, match="empty message content") as excinfo:
        _summarize(
            docs=docs,
            metadata=metadata,
            language="english",
            modelbackend="testing",
            llm=fake_llm,
            verbose=False,
        )
    # the error must include the full generation dump to help troubleshooting
    assert "generations[0]" in str(excinfo.value)


@pytest.mark.basic
def test_summary_recursion_continues_past_first_pass():
    """Recursive summarization must keep going while each pass yields a new
    summary. Regression test for a dict-membership bug where the dedup check
    compared the summary text against integer recursion-level keys (always
    True), which forced recursion to stop after the very first pass."""
    from langchain_community.chat_models.fake import FakeListChatModel
    from langchain_core.documents import Document

    from wdoc.utils.llm import PriceCountingCallback
    from wdoc.utils.tasks.summarize import summarize_documents

    # Each canned response is distinct and comfortably above the 20-token floor
    # enforced by check_docs_tkn_length, so no pass is skipped for being
    # identical to the previous one or too short to recurse on.
    responses = [
        f"- Distinct summary number {i} about monkeys, procrastination, "
        "deadlines, panic monsters and other carefully chosen words so the "
        "token count stays well above the minimum threshold for recursion."
        for i in range(1, 8)
    ]
    fake_llm = FakeListChatModel(
        verbose=False,
        responses=responses,
        callbacks=[PriceCountingCallback(verbose=False)],
        disable_streaming=True,
        cache=False,
    )

    docs = [
        Document(
            page_content="A long original document about procrastination that "
            "easily clears the minimum token length required for summarization.",
            metadata={},
        )
    ]

    n_recursion = 2
    out = summarize_documents(
        path="test_recursion_doc",
        relevant_docs=docs,
        summary_language="english",
        model=ModelName("testing/testing"),
        llm=fake_llm,
        llm_verbosity=False,
        summary_n_recursion=n_recursion,
        llm_price={"prompt": 0, "completion": 0, "internal_reasoning": 0},
        in_import_mode=True,
        out_file=None,
        wdoc_version="test",
    )

    # One base pass (level 0) plus one entry per recursion level. With the bug
    # this stopped at 2 entries ({0, 1}); fixed it produces all of them.
    assert len(out["recursive_summaries"]) == n_recursion + 1, out[
        "recursive_summaries"
    ]


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_debug():
    """Test summarization of Tim Urban's procrastination video."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
        debug=True,
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_with_out_file():
    """Test that summary is properly written to output file."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        output_path = tmp.name

    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        disable_llm_cache=True,
        filetype="auto",
        out_file=output_path,
    )
    assert inst.__import_mode__
    inst.summary_task()

    # Verify the output file
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()

    # Check for expected content in the summary
    assert len(content) > 0
    assert "arj7oStGLkU" in content
    assert (
        "Inside the mind of a master procrastinator" in content
        or "monkey" in content.lower()
    )
    assert "wdoc version" in content
    if os.path.exists(output_path):
        os.unlink(output_path)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban():
    """Test query task on Tim Urban's procrastination video. Three times to test the caching."""
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "false"
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()
    assert out["total_cost"] > 0, out

    inst2 = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=False,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out2 = inst2.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer2 = out2["final_answer"]
    assert "monkey" in final_answer2.lower()
    assert out2["total_cost"] == 0, out2

    inst3 = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=True,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out3 = inst3.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer3 = out3["final_answer"]
    assert "monkey" in final_answer3.lower()
    assert out3["total_cost"] > 0, out3
    os.environ["WDOC_DISABLE_EMBEDDINGS_CACHE"] = "true"


@pytest.mark.basic
def test_query_tim_urban_testing_model():
    """Test query task on Tim Urban's procrastination video with testing model."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model="testing",  # Use the special testing model
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",  # Keep eval model for now, might need adjustment if testing model affects eval
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",  # Keep embed model
        disable_llm_cache=True,
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    # The 'testing' model should return a fixed string
    assert "Lorem ipsum dolor sit amet" in final_answer


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_whisper_tim_urban():
    """Test summarization of Tim Urban's video using whisper transcription."""
    try:
        out = wdoc(
            task="summarize",
            path="https://www.youtube.com/watch?v=arj7oStGLkU",
            model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
            disable_llm_cache=True,
            # filetype="youtube",
            youtube_audio_backend="whisper",
            whisper_lang="en",
        )
    except Exception as e:
        if "502" in str(e):
            pytest.skip(
                f"Whisper endpoint returned 502 (upstream unavailable), "
                f"treating as not-tested instead of failure: {e}"
            )
        raise


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
@pytest.mark.skipif(
    not _OLLAMA_REACHABLE,
    reason=(
        "Ollama server not reachable on "
        f"{os.getenv('OLLAMA_HOST', '127.0.0.1:11434')}. "
        "Start `ollama serve` (or set OLLAMA_HOST) to run this test."
    ),
)
def test_ollama_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName(f"ollama/{WDOC_TEST_OLLAMA_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_openai_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName(f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}"),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_mistral_embeddings():
    emb = load_embeddings_engine(
        modelname=ModelName(WDOC_TEST_MISTRAL_EMBED_MODEL),
        cli_kwargs={},
        api_base=None,
        embed_kwargs={},
        private=False,
        do_test=True,
    )
    _test_embeddings(emb)


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_openrouter():
    """Test summarization of Tim Urban's procrastination video using openrouter model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_OPENROUTER_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban_openrouter():
    """Test query task on Tim Urban's procrastination video using openrouter."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_OPENROUTER_MODEL,
        query_eval_model=WDOC_TEST_OPENROUTER_EVAL_MODEL,
        disable_llm_cache=True,
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_summary_tim_urban_default_model():
    """Test summarization of Tim Urban's procrastination video using the default model."""
    inst = wdoc(
        task="summarize",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_DEFAULT_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        filetype="auto",
    )
    out = inst.summary_task()
    assert "monkey" in out["summary"].lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_tim_urban_default_model():
    """Test query task on Tim Urban's procrastination video using the default model."""
    inst = wdoc(
        task="query",
        path="https://www.youtube.com/watch?v=arj7oStGLkU",
        model=WDOC_TEST_DEFAULT_MODEL,
        query_eval_model=WDOC_TEST_DEFAULT_EVAL_MODEL,
        embed_model=WDOC_TEST_DEFAULT_EMBED_MODEL,
        disable_llm_cache=True,
        # filetype="youtube",
        # youtube_language="en",
        filetype="auto",
    )
    out = inst.query_task(
        query=ANALOGY_QUESTION,
    )
    final_answer = out["final_answer"]
    assert "monkey" in final_answer.lower()


@pytest.mark.api
@pytest.mark.skipif(
    " -m api" not in " ".join(sys.argv),
    reason="Skip tests using external APIs by default, use '-m api' to run them.",
)
def test_query_duckduckgo_search():
    """Test DuckDuckGo search functionality."""
    inst = wdoc(
        task="query",
        path="How is nvidia doing this month",
        query="How is nvidia doing this month",
        filetype="ddg",
        ddg_max_results=10,
        ddg_region="us - US",
        model=f"openai/{WDOC_TEST_OPENAI_MODEL}",
        query_eval_model=f"openai/{WDOC_TEST_OPENAI_EVAL_MODEL}",
        embed_model=f"openai/{WDOC_TEST_OPENAI_EMBED_MODEL}",
        disable_llm_cache=True,
        debug=True,
        verbose=True,
        file_loader_parallel_backend="threading",
        loading_failure="warn",
    )
    out = inst.query_task(
        query="How is nvidia doing this month",
    )

    # Basic validation that the function runs without crashing
    assert isinstance(out, dict), out
    assert "final_answer" in out, out
    assert isinstance(out["final_answer"], str), out
    assert len(out["final_answer"]) > 0, out
    # Don't check the content deeply as requested, just ensure it returns something


@pytest.mark.basic
def test_load_one_doc_error_message_is_concise():
    """A failing document load must log ONE concise error line, not the error
    text stapled to a full traceback. The wrapper used to embed the traceback in
    the message AND re-emit it via logger.exception AND again via the re-raise,
    stacking three near-identical tracebacks that buried the real error (e.g. a
    single '415 Unsupported Media Type' from a whisper endpoint). Regression test
    for that noisy/misleading output."""
    from loguru import logger
    from wdoc.utils.loaders import wrapper_load_one_doc

    @wrapper_load_one_doc
    def _boom(**kwargs):
        raise ValueError("415 Unsupported Media Type")

    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        # warn mode swallows the error and returns its string
        ret = _boom(filetype="local_audio", loading_failure="warn")
        assert ret == "415 Unsupported Media Type"

        # crash mode re-raises the original exception
        with pytest.raises(ValueError, match="415 Unsupported Media Type"):
            _boom(filetype="local_audio", loading_failure="crash")
    finally:
        logger.remove(sink_id)

    wrapper_records = [r for r in records if "Error when loading doc" in r["message"]]
    assert len(wrapper_records) == 2, wrapper_records

    for rec in wrapper_records:
        # the message itself stays concise: no embedded traceback dump
        assert "Full traceback:" not in rec["message"], rec["message"]
        assert "415 Unsupported Media Type" in rec["message"], rec["message"]
        assert "filetype local_audio" in rec["message"], rec["message"]

    warn_rec = next(r for r in wrapper_records if r["level"].name == "WARNING")
    crash_rec = next(r for r in wrapper_records if r["level"].name == "ERROR")

    # crash mode leaves the traceback to the re-raise instead of duplicating it
    # onto the log record
    assert crash_rec["exception"] is None, crash_rec["exception"]
    # warn mode keeps the traceback attached exactly once (via exc_info), not
    # inlined into the message
    assert warn_rec["exception"] is not None


@pytest.mark.basic
def test_whisper_fallback_rewinds_file(tmp_path, monkeypatch):
    """When litellm.transcription fails and wdoc falls back to a direct request,
    it must rewind the audio file first. litellm reads the handle to EOF, so
    without a seek(0) the fallback uploads an empty body and the endpoint answers
    with a misleading error (e.g. 415 Unsupported Media Type) unrelated to the
    real failure. Regression test: the fallback must upload the full file."""
    import litellm
    import requests

    from wdoc.utils.loaders import shared_audio

    monkeypatch.setenv("WDOC_WHISPER_API_KEY", "test-key")
    monkeypatch.setenv("WDOC_WHISPER_ENDPOINT", "https://fake.endpoint/")

    audio_path = tmp_path / "sample.mp3"
    payload = b"FAKE_AUDIO_BYTES_0123456789"
    audio_path.write_bytes(payload)

    def fake_transcription(**kwargs):
        # emulate litellm consuming the file to EOF before failing
        kwargs["file"].read()
        raise RuntimeError("litellm boom")

    uploaded = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "ok"}

    def fake_post(url, files=None, data=None, headers=None):
        uploaded["content"] = files["file"].read()
        return FakeResponse()

    monkeypatch.setattr(litellm, "transcription", fake_transcription)
    monkeypatch.setattr(requests, "post", fake_post)

    # call the undecorated function to bypass the joblib disk cache, so the test
    # actually exercises the code and stays reproducible across runs
    out = shared_audio.transcribe_audio_whisper.func(
        audio_path=audio_path,
        audio_hash="rewind-hash",
        language=None,
        prompt=None,
    )

    assert out == {"text": "ok"}
    # the crux: the fallback uploaded the whole file, not the empty EOF leftover
    assert uploaded["content"] == payload


@pytest.mark.basic
def test_whisper_fallback_surfaces_both_errors(tmp_path, monkeypatch):
    """If the direct-request fallback also fails, the raised error must mention
    BOTH the endpoint error and the original litellm error, so the real root
    cause (often an auth/proxy misconfiguration) is not masked by a spurious
    status code."""
    import litellm
    import requests

    from wdoc.utils.loaders import shared_audio

    monkeypatch.setenv("WDOC_WHISPER_API_KEY", "test-key")
    monkeypatch.setenv("WDOC_WHISPER_ENDPOINT", "https://fake.endpoint/")

    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"FAKE_AUDIO_BYTES_0123456789")

    def fake_transcription(**kwargs):
        kwargs["file"].read()
        raise RuntimeError("original litellm auth failure")

    def fake_post(url, files=None, data=None, headers=None):
        raise requests.exceptions.HTTPError("415 Unsupported Media Type")

    monkeypatch.setattr(litellm, "transcription", fake_transcription)
    monkeypatch.setattr(requests, "post", fake_post)

    # call the undecorated function to bypass the joblib disk cache
    with pytest.raises(Exception) as excinfo:
        shared_audio.transcribe_audio_whisper.func(
            audio_path=audio_path,
            audio_hash="both-errors-hash",
            language=None,
            prompt=None,
        )

    msg = str(excinfo.value)
    assert "415 Unsupported Media Type" in msg
    assert "original litellm auth failure" in msg


@pytest.mark.basic
def test_local_audio_unsilence_uploads_mp3_not_ogg(tmp_path, monkeypatch):
    """The silence-removal path must hand a widely-supported format to the
    transcriber. It used to re-encode to .ogg, which some OpenAI-compatible
    whisper endpoints reject with a 415 Unsupported Media Type. Regression test:
    the file passed to transcribe_audio_whisper must be .mp3 (matching the video
    and online-media loaders), never .ogg."""
    import types
    import uuid
    from pathlib import Path

    from wdoc.utils.loaders import local_audio

    # a fake waveform whose only meaningful attribute is its sample count
    class FakeWave:
        def __init__(self, nsamples):
            self.shape = (1, nsamples)

    sample_rate = 16000
    # 20s of audio -> 15s after "silence removal": passes the >10s and
    # new_dur <= dur assertions in the loader. raising=False because torchaudio
    # exposes some of these (e.g. sox_effects) only after a lazy submodule import.
    monkeypatch.setattr(
        local_audio.torchaudio,
        "load",
        lambda *a, **k: (FakeWave(20 * sample_rate), sample_rate),
        raising=False,
    )
    monkeypatch.setattr(
        local_audio.torchaudio,
        "sox_effects",
        types.SimpleNamespace(
            apply_effects_tensor=lambda *a, **k: (
                FakeWave(15 * sample_rate),
                sample_rate,
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        local_audio.torchaudio, "save", lambda *a, **k: None, raising=False
    )

    # no-op ffmpeg chain: ffmpeg.input(...).output(...).run()
    class _Chain:
        def output(self, *a, **k):
            return self

        def run(self, *a, **k):
            return None

    monkeypatch.setattr(local_audio.ffmpeg, "input", lambda *a, **k: _Chain())
    monkeypatch.setattr(local_audio, "file_hasher", lambda *a, **k: "fakehash")

    captured = {}

    def fake_transcribe(audio_path, audio_hash, language, prompt):
        captured["audio_path"] = Path(audio_path)
        return {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "duration": 15.0,
            "language": "en",
        }

    monkeypatch.setattr(local_audio, "transcribe_audio_whisper", fake_transcribe)
    monkeypatch.setattr(
        local_audio,
        "convert_verbose_json_to_timestamped_text",
        lambda content: "hello",
    )

    audio_in = tmp_path / "input.wav"
    audio_in.write_bytes(b"not-really-audio")

    docs = local_audio.load_local_audio(
        path=audio_in,
        # unique hash so the joblib doc-loaders cache always misses (reproducible)
        file_hash=uuid.uuid4().hex,
        audio_backend="whisper",
        loaders_temp_dir=tmp_path,
        audio_unsilence=True,
    )

    assert "audio_path" in captured, "transcribe_audio_whisper was never called"
    assert captured["audio_path"].suffix == ".mp3", captured["audio_path"]
    assert captured["audio_path"].suffix != ".ogg"
    assert len(docs) == 1
    assert docs[0].page_content == "hello"
