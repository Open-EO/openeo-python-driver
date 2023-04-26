
# OpenID Connect related setup and configuration



## Support for OIDC Client Credentials grant with user mapping


Experimental feature: support OIDC Client Credentials grant
through custom client-to-user mapping.

An access token obtained through OIDC Client Credentials grant
is associated with a client, but not directly with a user.
With custom user mapping in the config it is possible to map a client identifier
("sub" field in a JWT token or OIDC userinfo/introspection response)
to a user id.
At the moment this is just a dictionary based mapping,
where you map a tuple `(provider_id, sub)`
to a dictionary with at least a `user_id` field:

```python
config = OpenEoBackendConfig(
    ...,
    oidc_user_map={
        ("egi", "2b7c4f4c-3f4a-449d-8d61-54c9241208bc"): {"user_id": "john-doe"},
    },
)
```
