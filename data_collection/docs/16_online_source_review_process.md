# 16. Quy trình review nguồn online

## Mục lục
- [Trạng thái](#trạng-thái)
- [Quyết định](#quyết-định)
- [Checklist](#checklist)

## Trạng thái
`DISCOVERED` → `NEEDS_REVIEW` → một trong `APPROVED_FOR_DOWNLOAD`, `APPROVED_INTERNAL_ONLY`, `NEEDS_PERMISSION`, `LICENSE_UNVERIFIED`, `REJECTED`; sau đó mới `DOWNLOADED`/`PROCESSED`. Không chuyển `APPROVED_FOR_DOWNLOAD` nếu `license_verified` không là `TRUE`.

## Quyết định
Reviewer mở trang chính thức và trang license, lưu URL/evidence, kiểm tra quyền train/công bố/phân phối riêng. Có portal, thỏa thuận hay email cần xác minh thì dừng ở `NEEDS_PERMISSION`. URL trực tiếp phải do nhà cung cấp chính thức phát hành.

## Checklist
- [ ] Không copy URL từ nguồn thứ ba không chính thức.
- [ ] Không ghi PUBLIC DOMAIN/CC-BY/ALLOWED nếu chưa kiểm tra.
- [ ] Ghi rejected reason và mức rủi ro.
