const form = document.querySelector("#make");
if (form) {
  const source = document.querySelector("#image");
  const preview = document.querySelector("#preview");
  const photo = document.querySelector("#source-preview");
  const box = document.querySelector("#crop-box");
  const frame = document.querySelector("#frame");
  const x = document.querySelector("#focus_x");
  const y = document.querySelector("#focus_y");
  const zoom = document.querySelector("#zoom");
  const url = document.querySelector("#url");
  const code = document.querySelector("#source_qr");
  let previousURL;

  function positionCrop() {
    box.hidden = frame.value === "full";
    const width = photo.clientWidth;
    const height = photo.clientHeight;
    const side = Math.min(width, height) * Number(zoom.value) / 100;
    const left = Math.min(Math.max(0, width * Number(x.value) / 100 - side / 2), width - side);
    const top = Math.min(Math.max(0, height * Number(y.value) / 100 - side / 2), height - side);
    Object.assign(box.style, {
      width: side + "px", height: side + "px", left: left + "px", top: top + "px",
    });
  }
  source.addEventListener("change", () => {
    if (previousURL) URL.revokeObjectURL(previousURL);
    preview.hidden = !source.files.length;
    if (source.files.length) photo.src = previousURL = URL.createObjectURL(source.files[0]);
  });
  photo.addEventListener("load", positionCrop);
  new ResizeObserver(positionCrop).observe(photo);
  [frame, x, y, zoom].forEach((field) => field.addEventListener("input", positionCrop));
  preview.addEventListener("click", (event) => {
    frame.value = "crop";
    const bounds = photo.getBoundingClientRect();
    x.value = Math.round(100 * (event.clientX - bounds.left) / bounds.width);
    y.value = Math.round(100 * (event.clientY - bounds.top) / bounds.height);
    document.querySelector("#focus-details").open = true;
    positionCrop();
  });
  const layout = document.querySelector("#layout");
  const proportion = document.querySelector("#qr_percent");
  layout.addEventListener("input", () => {
    proportion.max = layout.value === "round" ? "66" : "80";
    if (Number(proportion.value) > Number(proportion.max)) proportion.value = proportion.max;
  });
  proportion.max = layout.value === "round" ? "66" : "80";

  const background = document.querySelector("#background");
  const surfaceType = document.querySelector("#scene_mount");
  const surfaceSetup = document.querySelector("#surface-setup");
  const scenePicker = document.querySelector("#scene-picker");
  const scenePhoto = document.querySelector("#scene-photo");
  const coordinates = document.querySelector("#scene_corners");
  const markers = document.querySelector("#scene-markers");
  let sceneURL;

  function setSurfaceMode() {
    const printing = surfaceType.value.startsWith("print-");
    surfaceSetup.hidden = !printing;
    document.querySelector("#scene-simple").hidden = printing;
    coordinates.required = printing;
    background.required = printing;
  }

  function showMarkers() {
    markers.replaceChildren();
    const points = coordinates.value.trim().split(";").filter(Boolean);
    points.slice(0, 4).forEach((point, index) => {
      const [left, top] = point.split(",").map(Number);
      if (![left, top].every(value => Number.isFinite(value) && value >= 0 && value <= 100)) return;
      const marker = document.createElement("span");
      marker.textContent = index + 1;
      marker.style.left = left + "%";
      marker.style.top = top + "%";
      markers.append(marker);
    });
  }

  background.addEventListener("change", () => {
    if (sceneURL) URL.revokeObjectURL(sceneURL);
    scenePicker.hidden = !background.files.length;
    if (background.files.length) scenePhoto.src = sceneURL = URL.createObjectURL(background.files[0]);
  });
  scenePicker.addEventListener("click", event => {
    const bounds = scenePhoto.getBoundingClientRect();
    const point = [100 * (event.clientX - bounds.left) / bounds.width,
                   100 * (event.clientY - bounds.top) / bounds.height].map(n => n.toFixed(2));
    const points = coordinates.value.trim().split(";").filter(Boolean);
    coordinates.value = [...(points.length < 4 ? points : []), point.join(",")].join(";");
    showMarkers();
  });
  coordinates.addEventListener("input", showMarkers);
  surfaceType.addEventListener("input", setSurfaceMode);
  document.querySelector("#scene-reset").addEventListener("click", () => {
    coordinates.value = "";
    showMarkers();
  });
  showMarkers();
  setSurfaceMode();

  function checkSource() {
    url.setCustomValidity(url.value && code.files.length ? "请只填写链接或上传原码中的一种。" : "");
  }
  [url, code].forEach((field) => field.addEventListener("input", checkSource));
  form.addEventListener("submit", (event) => {
    if (!url.value && !code.files.length) {
      event.preventDefault();
      url.setCustomValidity("请输入链接或上传原码。");
      url.reportValidity();
      return;
    }
    document.querySelector("#busy").textContent = "正在生成设计并读取整图…";
    form.querySelector("button[type=submit]").disabled = true;
  });
}
