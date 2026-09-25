"""
Exercises the actual HTTP layer (app/main.py) with FastAPI's TestClient —
no live server, no network, fully deterministic. This is the closest thing
in the test suite to what the capstone brief's acceptance probes do against
a live system (§12, Probes 2-6).
"""
from __future__ import annotations

from app.models import Image, ImageEmbedding, ImageStatus, Post, PostEmbedding

MODEL = "test-embedding-model"


def _seed(session_factory):
    """Seed a fox post + a matching fox image + a mismatched wolf image."""
    session = session_factory()
    try:
        post = Post(slug="fox-post", title="The Secret Life of Red Foxes", body_text="About the red fox (Vulpes vulpes).")
        session.add(post)
        session.flush()
        session.add(PostEmbedding(post_id=post.id, model_name=MODEL, vector=[1.0, 0.0]))

        fox = Image(
            filename="fox.jpg", filepath="/tmp/fox.jpg", subject="red fox", category="animal",
            confidence=0.9, status=ImageStatus.TAGGED, attributes=["orange"], caption="A red fox",
        )
        session.add(fox)
        session.flush()
        session.add(ImageEmbedding(image_id=fox.id, model_name=MODEL, vector=[0.9, 0.1]))

        wolf = Image(
            filename="wolf.jpg", filepath="/tmp/wolf.jpg", subject="gray wolf", category="animal",
            confidence=0.9, status=ImageStatus.TAGGED, attributes=["gray"], caption="A gray wolf",
        )
        session.add(wolf)
        session.flush()
        session.add(ImageEmbedding(image_id=wolf.id, model_name=MODEL, vector=[0.95, 0.05]))  # ranks ABOVE fox

        no_match_post = Post(slug="lakes-post", title="Mountain Lakes", body_text="About still alpine lakes at sunrise.")
        session.add(no_match_post)
        session.flush()
        session.add(PostEmbedding(post_id=no_match_post.id, model_name=MODEL, vector=[0.0, 1.0]))

        session.commit()
        return {"post_id": post.id, "post_slug": post.slug, "fox_id": fox.id, "wolf_id": wolf.id, "no_match_post_id": no_match_post.id}
    finally:
        session.close()


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_posts(api_client):
    ids = _seed(api_client.test_sessionmaker)
    resp = api_client.get("/posts")
    assert resp.status_code == 200
    slugs = {p["slug"] for p in resp.json()}
    assert "fox-post" in slugs and "lakes-post" in slugs


def test_get_post_images_404_for_unknown_post(api_client):
    resp = api_client.get("/posts/does-not-exist/images")
    assert resp.status_code == 404


# --- Probe 2: fox post -> fox ranks first, wolf/dog rank lower ------------

def test_probe2_fox_post_surfaces_fox_first(api_client):
    ids = _seed(api_client.test_sessionmaker)

    resp = api_client.get(f"/posts/{ids['post_slug']}/images")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "matched"
    assert body["best"]["image"]["id"] == ids["fox_id"]
    assert body["best"]["image"]["subject"] == "red fox"


# --- Probe 3: wolf rejected with a category-mismatch explanation, EVEN --
#     though it scores higher than the fox on raw similarity -------------

def test_probe3_wolf_visible_among_candidates_and_explicitly_rejected(api_client):
    ids = _seed(api_client.test_sessionmaker)

    resp = api_client.get(f"/posts/{ids['post_slug']}/images?include_candidates=true")
    body = resp.json()

    wolf_candidate = next(c for c in body["candidates"] if c["image"]["id"] == ids["wolf_id"])
    assert wolf_candidate["guard_status"] == "rejected"
    assert "mismatch" in wolf_candidate["guard_reason"].lower()
    assert "wolf" in wolf_candidate["guard_reason"].lower()


# --- Probe 4: a post with no suitable image -> no confident match --------

def test_probe4_no_confident_match_with_reasons(api_client):
    ids = _seed(api_client.test_sessionmaker)

    resp = api_client.get(f"/posts/{ids['no_match_post_id']}/images")
    body = resp.json()

    assert body["status"] == "no_confident_match"
    assert body["best"] is None
    assert len(body["explanation"]) > 0


# --- The review workflow: approve / reject / inspect ----------------------

def test_review_workflow_approve_then_inspect(api_client):
    ids = _seed(api_client.test_sessionmaker)

    images_resp = api_client.get(f"/posts/{ids['post_slug']}/images")
    suggestion_id = images_resp.json()["best"]["id"]

    review_resp = api_client.post(
        f"/suggestions/{suggestion_id}/review",
        json={"decision": "approve", "reviewer": "demo-user", "notes": "looks right"},
    )
    assert review_resp.status_code == 201
    assert review_resp.json()["decision"] == "approve"

    inspect_resp = api_client.get(f"/suggestions/{suggestion_id}")
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["review"]["decision"] == "approve"
    assert inspect_resp.json()["review"]["notes"] == "looks right"


def test_review_workflow_reject_the_wolf(api_client):
    ids = _seed(api_client.test_sessionmaker)

    images_resp = api_client.get(f"/posts/{ids['post_slug']}/images")
    wolf_suggestion = next(
        c for c in images_resp.json()["candidates"] if c["image"]["id"] == ids["wolf_id"]
    )

    resp = api_client.post(f"/suggestions/{wolf_suggestion['id']}/review", json={"decision": "reject"})
    assert resp.status_code == 201
    assert resp.json()["decision"] == "reject"
    assert resp.json()["reviewer"] == "local-reviewer"  # default applied


def test_double_review_returns_409(api_client):
    ids = _seed(api_client.test_sessionmaker)
    images_resp = api_client.get(f"/posts/{ids['post_slug']}/images")
    suggestion_id = images_resp.json()["best"]["id"]

    first = api_client.post(f"/suggestions/{suggestion_id}/review", json={"decision": "approve"})
    assert first.status_code == 201

    second = api_client.post(f"/suggestions/{suggestion_id}/review", json={"decision": "reject"})
    assert second.status_code == 409


def test_review_unknown_suggestion_returns_404(api_client):
    resp = api_client.post("/suggestions/does-not-exist/review", json={"decision": "approve"})
    assert resp.status_code == 404


def test_review_invalid_decision_returns_422(api_client):
    ids = _seed(api_client.test_sessionmaker)
    images_resp = api_client.get(f"/posts/{ids['post_slug']}/images")
    suggestion_id = images_resp.json()["best"]["id"]

    resp = api_client.post(f"/suggestions/{suggestion_id}/review", json={"decision": "maybe"})
    assert resp.status_code == 422  # validation at the boundary — never reaches app/review.py


# --- Probe 6: cost log is queryable ----------------------------------------

def test_cost_summary_endpoint(api_client):
    resp = api_client.get("/costs/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_estimated_cost_usd" in body
    assert "by_call_type" in body


# --- Idempotency: repeat GET does not create duplicate suggestion rows ----

def test_repeated_get_post_images_does_not_duplicate_suggestions(api_client):
    ids = _seed(api_client.test_sessionmaker)

    first = api_client.get(f"/posts/{ids['post_slug']}/images").json()
    second = api_client.get(f"/posts/{ids['post_slug']}/images").json()

    assert len(first["candidates"]) == len(second["candidates"]) == 2
    assert {c["id"] for c in first["candidates"]} == {c["id"] for c in second["candidates"]}