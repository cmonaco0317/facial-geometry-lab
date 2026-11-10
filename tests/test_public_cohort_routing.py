import unittest

from local_app.cohorts import CohortResolver
from local_app.contracts import Demographics


class PublicCohortRoutingTests(unittest.TestCase):
    def inventory(self):
        return {
            "m_white_60_69": dict(n=180, sex="male", ethnicity="white",
                                  age_band="60_69", meets_floor=False),
            "m_white_60_69_to_70_plus": dict(
                n=220, sex="male", ethnicity="white", age_band="60_69_to_70_plus",
                meets_floor=True, merged=True,
                members=["m_white_60_69", "m_white_70_plus"],
            ),
            "m_black_60_69": dict(n=180, sex="male", ethnicity="black",
                                  age_band="60_69", meets_floor=False),
        }

    def test_suppressed_cells_cannot_create_a_fake_materialized_pool(self):
        resolver = CohortResolver(self.inventory(), floor=150, allow_virtual_pools=False)
        choice = resolver.resolve(Demographics(65, "male", ("white",)))
        self.assertEqual(choice.used_id, "m_white_60_69_to_70_plus")
        self.assertNotEqual(choice.requested_id, choice.used_id)
        self.assertEqual(choice.member_ids, ("m_white_60_69", "m_white_70_plus"))

    def test_existing_generic_resolver_keeps_explicit_virtual_pool_option(self):
        resolver = CohortResolver(self.inventory(), floor=150)
        self.assertEqual(resolver.resolve(Demographics(65, "male", ("white",))).used_id,
                         "m_all_ethnicities_60_69")

    def test_unsupported_reference_is_explained_before_photo_storage(self):
        from unittest.mock import patch
        from local_app.cohorts import NoCompatibleCohort
        from tests.test_server import LocalServerTests, multipart, jpeg_bytes

        fixture = LocalServerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        token = fixture._token(fixture._landing())
        body, content_type = multipart(
            [("csrf_token", token), ("age", "30"), ("sex", "another"),
             ("ethnicity", "another_or_not_listed")],
            [("photos", "synthetic.jpg", "image/jpeg", jpeg_bytes(50))],
        )
        with patch.object(fixture.server.analysis.resolver, "resolve",
                          side_effect=NoCompatibleCohort("unavailable")):
            with patch.object(fixture.server.store, "create",
                              wraps=fixture.server.store.create) as create:
                status, _, response = fixture.client.request(
                    "POST", "/sessions", body=body,
                    headers={"Content-Type": content_type,
                             "Origin": f"http://127.0.0.1:{fixture.server.server_address[1]}"},
                )
        self.assertEqual(status, 422)
        self.assertIn(b"reference", response.lower())
        create.assert_not_called()
        self.assertFalse(fixture.server.store.has_owned_sessions)

    def test_existing_unsupported_session_is_not_deleted_or_extended(self):
        from unittest.mock import patch
        from local_app.cohorts import NoCompatibleCohort
        from local_app.photos import PhotoCatalog
        from tests.test_server import LocalServerTests, multipart, jpeg_bytes

        fixture = LocalServerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        session_id, token = fixture._create_session(photo_count=1)
        before = PhotoCatalog(fixture.server.store, session_id).entries()
        body, content_type = multipart([("csrf_token", token)],
            [("photos", "synthetic-new.jpg", "image/jpeg", jpeg_bytes(80))])
        with patch.object(fixture.server.analysis.resolver, "resolve",
                          side_effect=NoCompatibleCohort("unavailable")):
            status, _, _ = fixture.client.request("POST", f"/sessions/{session_id}/photos",
                body=body, headers={"Content-Type": content_type,
                "Origin": f"http://127.0.0.1:{fixture.server.server_address[1]}"})
            self.assertEqual(status, 422)
            status, _, _ = fixture.client.request("GET", f"/sessions/{session_id}/results")
            self.assertEqual(status, 422)
        self.assertEqual(PhotoCatalog(fixture.server.store, session_id).entries(), before)
        self.assertTrue(fixture.server.store.load(session_id).root.exists())

    def test_unsupported_archive_is_rejected_before_restoring_photos(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from local_app.cohorts import NoCompatibleCohort
        from local_app.history import ArchiveError
        from local_app.history_routes import restore
        from tests.test_server import LocalServerTests

        fixture = LocalServerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        record = SimpleNamespace(evidence=[object()], snapshot=SimpleNamespace(
            demographics=SimpleNamespace(age=30, sex="another", ethnicities=["white"])))
        with patch.object(fixture.server.analysis.resolver, "resolve",
                          side_effect=NoCompatibleCohort("unavailable")):
            with patch.object(fixture.server.store, "create") as create:
                with self.assertRaisesRegex(ArchiveError, "archive is unchanged"):
                    restore(SimpleNamespace(server=fixture.server), record)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
