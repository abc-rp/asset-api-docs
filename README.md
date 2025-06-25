[![tests](https://github.com/abc-rp/asset-api-docs/actions/workflows/tests.yaml/badge.svg)](https://github.com/abc-rp/asset-api-docs/actions/workflows/tests.yaml) [![format](https://github.com/abc-rp/asset-api-docs/actions/workflows/format.yaml/badge.svg)](https://github.com/abc-rp/asset-api-docs/actions/workflows/format.yaml)

# xRI Asset API (didapi.io)

The **xRI Asset API** allows users to retrieve assets and insights generated from Built Environment Scanning System (BESS) data capture. The API provides secure, authenticated access to binary asset files using a unique identifier and file extension.

## Base URL

```
https://didapi.io/v1
```

## Authentication

All requests must include an API key in the `x-api-key` header for authentication:

```http
x-api-key: YOUR_API_KEY_HERE
```

For the time being, xRI will generate keys on your behalf and will be sent to you via email. Contact support for any API key related questions.

## Endpoint

### Retrieve an Asset

**GET** `/result/{resultUuid}.{ext}`

#### Path Parameters:
- `resultUuid` (string, required): Unique identifier for the asset.
- `ext` (string, required): File extension of the asset (e.g., `png`, `webp`).

#### Headers:
- `x-api-key` (string, required): Your API authentication key.

#### Example Request

```bash
curl -X GET "https://didapi.io/v1/result/123e4567-e89b-12d3-a456-426614174000.png" \
-H "x-api-key: YOUR_API_KEY_HERE" --output asset.png
```

This will download the asset as `asset.png`.

## Responses

### Success Response
- **Status:** `200 OK`
- **Body:** Binary content of the requested asset.

### Error Responses

| Status Code | Error                      | Description                                            |
|-------------|----------------------------|--------------------------------------------------------|
| `401`       | Unauthorized               | Missing or invalid API key.                            |
| `404`       | Asset not found            | The requested asset does not exist.                    |
| `429`       | Rate limit exceeded        | Too many requests. Please wait and retry later.        |
| `500`       | Internal server error      | An unexpected error occurred while processing request. |

### Error Example

```json
{
  "error": "Asset not found."
}
```

## API Schema

A full OpenAPI schema can be found in this repository with the file extension `oas.json`.

## Rate Limiting

The API implements rate limiting. If your rate limit is exceeded, you'll receive a `429` error. Please contact support to request an increase in your quota if needed.

## Notes
- Ensure your API key is kept secure and never exposed publicly.
- API keys are not stored by xRI on disk. Once a key is generated make sure you store this securely as it cannot be retrieved again.
- Contact support for assistance or to resolve quota-related issues.
