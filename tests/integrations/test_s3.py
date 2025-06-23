import boto3
from botocore.exceptions import ParamValidationError

from openeo_driver.integrations.s3.client import S3ClientBuilder
import pytest

from openeo_driver.integrations.s3.credentials import get_credentials
from urllib import parse

from openeo_driver.integrations.s3.presigned_url import create_presigned_url

# Without protocol because legacy eodata config did not have protocol in endpoint
eodata_test_endpoint = "eodata.oeo.test"
legacy_test_endpoint = "https://customs3.oeo.test"


@pytest.fixture
def legacy_fallback_endpoint(monkeypatch):
    monkeypatch.setenv("SWIFT_URL", legacy_test_endpoint)


@pytest.fixture
def historic_eodata_endpoint_env_config(monkeypatch):
    monkeypatch.setenv("AWS_S3_ENDPOINT", eodata_test_endpoint)
    monkeypatch.setenv("AWS_HTTPS", "NO")


@pytest.fixture
def new_eodata_endpoint_env_config(monkeypatch):
    monkeypatch.setenv("EODATA_S3_ENDPOINT", f"https://{eodata_test_endpoint}")


@pytest.fixture
def swift_credentials(monkeypatch):
    monkeypatch.setenv("SWIFT_ACCESS_KEY_ID", "swiftkey")
    monkeypatch.setenv("SWIFT_SECRET_ACCESS_KEY", "swiftsecret")


@pytest.mark.parametrize(
    ["region_name", "expected_endpoint"],
    [
        ("waw3-1", "https://s3.waw3-1.cloudferro.com"),
        ("waw3-2", "https://s3.waw3-2.cloudferro.com"),
        ("waw4-1", "https://s3.waw4-1.cloudferro.com"),
        ("eu-nl", "https://obs.eu-nl.otc.t-systems.com"),
        ("EU-NL", "https://obs.EU-NL.otc.t-systems.com"),
        ("eu-de", "https://obs.eu-de.otc.t-systems.com"),
        ("eu-faketest-central", legacy_test_endpoint),
    ],
)
def test_s3_client_has_expected_endpoint_and_region(
    historic_eodata_endpoint_env_config,
    swift_credentials,
    legacy_fallback_endpoint,
    region_name: str,
    expected_endpoint: str,
):
    c = S3ClientBuilder.from_region(region_name)
    assert region_name == c.meta.region_name
    assert expected_endpoint == c.meta.endpoint_url


@pytest.mark.parametrize(
    "region_name,provider_name,env,exp_akid,exp_secret",
    [
        pytest.param(
            "waw3-1",
            "cf",
            {
                "SWIFT_ACCESS_KEY_ID": "swiftkey",
                "SWIFT_SECRET_ACCESS_KEY": "swiftsecret",
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "swiftkey",
            "swiftsecret",
            id="backwardsCompatibleSwiftFallbackButBeforeAWS",
        ),
        pytest.param(
            "waw3-1",
            "cf",
            {
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "awskey",
            "awssecret",
            id="backwardsCompatibleFallbackAWS",
        ),
        pytest.param(
            "waw3-1",
            "cf",
            {
                "OTC_ACCESS_KEY_ID": "cfaccess",
                "OTC_SECRET_ACCESS_KEY": "csecret",
                "SWIFT_ACCESS_KEY_ID": "swiftkey",
                "SWIFT_SECRET_ACCESS_KEY": "swiftsecret",
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "swiftkey",
            "swiftsecret",
            id="VendorSpecificCredentialsTakePrecedenceButIgnoredIfNotRightVendor",
        ),
        pytest.param(
            "waw3-1",
            "cf",
            {
                "CF_ACCESS_KEY_ID": "cfaccess",
                "CF_SECRET_ACCESS_KEY": "csecret",
                "SWIFT_ACCESS_KEY_ID": "swiftkey",
                "SWIFT_SECRET_ACCESS_KEY": "swiftsecret",
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "cfaccess",
            "csecret",
            id="VendorSpecificCredentialsTakePrecedenceIfMatch",
        ),
        pytest.param(
            "waw3-1",
            "cf",
            {
                "WAW3_1_ACCESS_KEY_ID": "cfaccesswaw31",
                "WAW3_1_SECRET_ACCESS_KEY": "csecretwaw31",
                "CF_ACCESS_KEY_ID": "cfaccess",
                "CF_SECRET_ACCESS_KEY": "csecret",
                "SWIFT_ACCESS_KEY_ID": "swiftkey",
                "SWIFT_SECRET_ACCESS_KEY": "swiftsecret",
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "cfaccesswaw31",
            "csecretwaw31",
            id="RegionSpecificCredentialsTakePrecedenceIfMatch",
        ),
        pytest.param(
            "eu-nl",
            "otc",
            {
                "WAW3_1_ACCESS_KEY_ID": "cfaccesswaw31",
                "WAW3_1_SECRET_ACCESS_KEY": "csecretwaw31",
                "OTC_ACCESS_KEY_ID": "otcaccess",
                "OTC_SECRET_ACCESS_KEY": "otcsecret",
                "SWIFT_ACCESS_KEY_ID": "swiftkey",
                "SWIFT_SECRET_ACCESS_KEY": "swiftsecret",
                "AWS_ACCESS_KEY_ID": "awskey",
                "AWS_SECRET_ACCESS_KEY": "awssecret",
            },
            "otcaccess",
            "otcsecret",
            id="RegionSpecificCredentialsIgnoredIfNoMatch",
        ),
    ],
)
def test_s3_credentials_retrieval_from_env(
    monkeypatch, region_name: str, provider_name: str, env: dict, exp_akid, exp_secret: str
):
    for env_var, env_val in env.items():
        monkeypatch.setenv(env_var, env_val)
    creds = get_credentials(region_name, provider_name)
    assert exp_akid == creds["aws_access_key_id"]
    assert exp_secret == creds["aws_secret_access_key"]


def test_exception_when_not_having_legacy_config_and_unsupported_region(
    historic_eodata_endpoint_env_config, swift_credentials
):
    with pytest.raises(EnvironmentError):
        S3ClientBuilder.from_region("eu-faketest-central")


@pytest.fixture(scope="function")
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")


def test_presigning_get_object_with_custom_parameter(aws_credentials):
    # GIVEN a supported custom parameter with any value
    test_key = "X-Proxy-Head-As-Get"
    test_value = "true"
    # GIVEN an S3 client
    s3 = boto3.client("s3")
    # WHEN we generate a presigned url requesting extra parameters

    u = create_presigned_url(s3, "test-bucket", "test_object", parameters={test_key: test_value})
    # THEN the value is passed in the url
    u_qs = parse.parse_qs(parse.urlsplit(u).query)
    assert u_qs[test_key] == [test_value]


def test_presigning_get_object_with_unsupported_custom_parameter(aws_credentials):
    # GIVEN an unsupported custom parameter with any value
    test_key = "unsupported-test"
    test_value = "value123-"
    # GIVEN an S3 client
    s3 = boto3.client("s3")
    # WHEN we generate a presigned url requesting extra parameters

    # THEN parameter validation should fail
    with pytest.raises(ParamValidationError):
        create_presigned_url(s3, "test-bucket", "test_object", parameters={test_key: test_value})
