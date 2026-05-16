import base64
import io
from pathlib import Path
from threading import Lock


CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented",
]

CLASS_DESCRIPTIONS = {
    "Non-Demented": "No signs of cognitive decline detected. Brain structures appear within normal range.",
    "Very Mild Demented": "Very early-stage indicators present. Subtle changes in memory-related regions.",
    "Mild Demented": "Mild atrophy observed. Hippocampal and cortical changes consistent with MCI.",
    "Moderate Demented": "Moderate neurodegeneration detected. Significant structural changes are visible.",
}

CLASS_COLORS = {
    "Non-Demented": "#34d399",
    "Very Mild Demented": "#fbbf24",
    "Mild Demented": "#fb923c",
    "Moderate Demented": "#f87171",
}


class MRIAnalysisError(RuntimeError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


class AlzheimerMRIService:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "Models" / "alzheimer_mri"
        self.model_candidates = [
            self.model_dir / "phase1_best.keras",
            self.model_dir / "alzheimer_model.h5",
        ]
        self._model = None
        self._model_path = None
        self._model_lock = Lock()
        self._deps_loaded = False

    def analyze(self, file_storage, lime_samples=300, lime_features=5, include_lime=True):
        self._load_dependencies()

        image = self._open_image(file_storage)
        if not self._is_likely_mri(image):
            raise MRIAnalysisError(
                "The uploaded image does not appear to be a grayscale axial brain MRI.",
                status_code=400,
            )

        model = self._load_model()
        processed_image = self._preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        probabilities = self._normalize_predictions(predictions)
        pred_idx = int(self.np.argmax(probabilities))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probabilities[pred_idx])

        heatmap = self._make_gradcam_heatmap(processed_image, model, pred_idx)
        gradcam_image = self._overlay_gradcam(image, heatmap)

        lime_image = None
        lime_warning = None
        if include_lime:
            try:
                lime_image = self._generate_lime_explanation(image, lime_samples, lime_features)
            except Exception as exc:
                lime_warning = f"LIME explanation could not be generated: {exc}"

        return {
            "prediction": {
                "class_name": pred_class,
                "description": CLASS_DESCRIPTIONS[pred_class],
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 1),
                "stage_index": pred_idx + 1,
                "stage_total": len(CLASS_NAMES),
                "severity": ["None", "Low", "Moderate", "High"][pred_idx],
                "color": CLASS_COLORS[pred_class],
            },
            "probabilities": [
                {
                    "class_name": class_name,
                    "probability": float(probabilities[index]),
                    "percent": round(float(probabilities[index]) * 100, 1),
                    "color": CLASS_COLORS[class_name],
                }
                for index, class_name in enumerate(CLASS_NAMES)
            ],
            "images": {
                "gradcam": self._array_to_data_uri(gradcam_image),
                "lime": self._array_to_data_uri(lime_image) if lime_image is not None else None,
            },
            "model_file": self._model_path.name if self._model_path else None,
            "lime_warning": lime_warning,
        }

    def _load_dependencies(self):
        if self._deps_loaded:
            return

        try:
            import cv2
            import numpy as np
            import tensorflow as tf
            from lime import lime_image
            from PIL import Image, UnidentifiedImageError
            from skimage.segmentation import mark_boundaries
        except ImportError as exc:
            raise MRIAnalysisError(
                f"Missing MRI analysis dependency: {exc.name}. Install the MRI feature requirements and restart the app.",
                status_code=503,
            ) from exc

        self.cv2 = cv2
        self.np = np
        self.tf = tf
        self.lime_image = lime_image
        self.Image = Image
        self.UnidentifiedImageError = UnidentifiedImageError
        self.mark_boundaries = mark_boundaries
        self._deps_loaded = True

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            for model_path in self.model_candidates:
                if not model_path.exists():
                    continue
                try:
                    self._model = self.tf.keras.models.load_model(str(model_path), compile=False)
                    self._model_path = model_path
                    return self._model
                except Exception as exc:
                    last_error = exc

            if any(path.exists() for path in self.model_candidates):
                raise MRIAnalysisError(f"Could not load the Alzheimer MRI model: {last_error}", status_code=500)

            raise MRIAnalysisError(
                f"No Alzheimer MRI model file was found in {self.model_dir}.",
                status_code=500,
            )

    def _open_image(self, file_storage):
        if file_storage is None or not getattr(file_storage, "filename", ""):
            raise MRIAnalysisError("No MRI image file was provided.", status_code=400)

        try:
            image = self.Image.open(file_storage.stream)
            image.load()
            return image.convert("RGB")
        except self.UnidentifiedImageError as exc:
            raise MRIAnalysisError("The uploaded file is not a readable image.", status_code=400) from exc

    def _is_likely_mri(self, image):
        gray = self.np.array(image.convert("L"))
        mean_intensity = gray.mean()
        std_intensity = gray.std()
        return mean_intensity < 200 and std_intensity > 10

    def _preprocess_image(self, image):
        resized = image.convert("RGB").resize((224, 224))
        image_array = self.np.array(resized, dtype="float32") / 255.0
        return self.np.expand_dims(image_array, axis=0)

    def _normalize_predictions(self, predictions):
        probabilities = self.np.asarray(predictions[0], dtype="float32").flatten()
        if probabilities.size != len(CLASS_NAMES):
            raise MRIAnalysisError(
                f"The loaded MRI model returned {probabilities.size} classes; expected {len(CLASS_NAMES)}.",
                status_code=500,
            )
        return probabilities

    def _make_gradcam_heatmap(self, image_array, model, pred_idx):
        layer_name = self._resolve_gradcam_layer_name(model)
        last_conv_layer = model.get_layer(layer_name)
        grad_model = self.tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output],
        )

        with self.tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            class_channel = predictions[:, pred_idx]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = self.tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = self.tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
        heatmap = self.tf.maximum(heatmap, 0)
        heatmap /= self.tf.reduce_max(heatmap) + 1e-8
        return heatmap.numpy()

    def _resolve_gradcam_layer_name(self, model):
        try:
            model.get_layer("top_conv")
            return "top_conv"
        except ValueError:
            pass

        for layer in reversed(model.layers):
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is None and hasattr(layer, "output"):
                output_shape = layer.output.shape
            if output_shape is not None and len(output_shape) == 4:
                return layer.name

        raise MRIAnalysisError("No convolutional layer was found for Grad-CAM.", status_code=500)

    def _overlay_gradcam(self, image, heatmap):
        heatmap_resized = self.cv2.resize(heatmap, (224, 224))
        heatmap_colored = self.cv2.applyColorMap(
            self.np.uint8(255 * heatmap_resized),
            self.cv2.COLORMAP_TURBO,
        )
        heatmap_colored = self.cv2.cvtColor(heatmap_colored, self.cv2.COLOR_BGR2RGB)
        image_rgb = self.np.array(image.convert("RGB").resize((224, 224))).astype("float32")
        overlay = heatmap_colored * 0.65 + image_rgb * 0.6
        return self.np.clip(overlay, 0, 255).astype("uint8")

    def _lime_predict(self, images):
        batch = self.np.array(images, dtype="float32") / 255.0
        return self._load_model().predict(batch, verbose=0)

    def _generate_lime_explanation(self, image, lime_samples, lime_features):
        explainer = self.lime_image.LimeImageExplainer()
        image_array = self.np.array(image.convert("RGB").resize((224, 224)))
        explanation = explainer.explain_instance(
            image_array,
            self._lime_predict,
            top_labels=1,
            hide_color=0,
            num_samples=lime_samples,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=lime_features,
            hide_rest=False,
        )
        return self.mark_boundaries(temp / 255.0, mask)

    def _array_to_data_uri(self, image_array):
        array = self.np.asarray(image_array)
        if array.dtype != self.np.uint8:
            max_value = float(array.max()) if array.size else 1.0
            if max_value <= 1.0:
                array = array * 255.0
            array = self.np.clip(array, 0, 255).astype("uint8")

        image = self.Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


_alzheimers_mri_service = None


def get_alzheimers_mri_service():
    global _alzheimers_mri_service
    if _alzheimers_mri_service is None:
        _alzheimers_mri_service = AlzheimerMRIService()
    return _alzheimers_mri_service
