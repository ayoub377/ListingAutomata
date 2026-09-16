# Meta Catalog bulk formats

Two output files, same data. Field names differ slightly between the Batch API (JSON) and the feed file (CSV) — use exactly the names below.

## Required fields (both formats)

| Concept | JSON (Batch API) | CSV feed | Notes |
|---|---|---|---|
| Unique ID | `retailer_id` | `id` | Stable slug, max 100 chars, no spaces. Reused on updates. |
| Title | `name` | `title` | ≤ 200 chars hard limit; keep ≤ 65 for display. |
| Description | `description` | `description` | ≤ 9999 chars; keep it short. |
| Availability | `availability` | `availability` | `in stock`, `out of stock`, `preorder`, `available for order`, `discontinued`. Default `in stock` unless told otherwise. |
| Condition | `condition` | `condition` | `new`, `refurbished`, `used`. Default `new`. |
| Price | `price` | `price` | JSON: integer in **cents/centimes** + separate `currency` field (e.g. `price: 34900, currency: "MAD"`). CSV: string with currency code, e.g. `349.00 MAD`. |
| Product page | `url` | `link` | Must be a valid URL. Placeholder if user has no site — flag it. |
| Main image | `image_url` | `image_link` | Public direct-image URL only. Default: Google Drive upload → `https://lh3.googleusercontent.com/d/<FILE_ID>` (verified by fetching). Fall back to `UPLOAD://` placeholders if hosting isn't possible. |
| Brand | `brand` | `brand` | From identification; omit if genuinely unbranded (CSV: leave empty). |

## Recommended optional fields

- `additional_image_url` (JSON) / `additional_image_link` (CSV): the two generated angle shots. CSV takes comma-separated URLs in one cell.
- `google_product_category` or `fb_product_category`: improves ad delivery; use the closest standard category string (e.g. `Sporting Goods > Outdoor Recreation > Camping & Hiking > Tents`).
- `sale_price` (same format as price) if the user gives a promo price.
- `inventory`: integer stock count if provided.

## JSON batch payload shape

```json
{
  "allow_upsert": true,
  "requests": [
    {
      "method": "CREATE",
      "retailer_id": "naturehike-cloud-up-2",
      "data": {
        "name": "Naturehike Cloud Up 2 — Ultralight 2-Person Tent",
        "description": "…",
        "availability": "in stock",
        "condition": "new",
        "price": 149900,
        "currency": "MAD",
        "url": "https://example.com/naturehike-cloud-up-2",
        "image_url": "UPLOAD://product-01/01-main.jpg",
        "additional_image_urls": ["UPLOAD://product-01/02-angle.jpg", "UPLOAD://product-01/03-context.jpg"],
        "brand": "Naturehike"
      }
    }
  ]
}
```

This payload is sent to `POST /{catalog_id}/items_batch` (`item_type=PRODUCT_ITEM`) with an access token — the user does that part, or a future automation does. The skill only writes the file.

## CSV feed shape

Header row exactly:

```
id,title,description,availability,condition,price,link,image_link,additional_image_link,brand,google_product_category
```

One row per product. Quote any field containing commas. UTF-8, no BOM. This file uploads directly in Commerce Manager → Catalog → Data sources → Data feed → Upload file.

## Validation before writing

- Every `id` unique and slug-safe (`[a-z0-9-]`).
- Every price parses (or is the literal `PRICE_TBD`, which the summary must flag).
- Titles ≤ 200 chars, no leading/trailing whitespace.
- Availability/condition values are from the allowed lists verbatim (lowercase).
- Count of rows in CSV == count of requests in JSON == count of product folders.
