import React, { useState, useRef } from "react";

const ImageToDataUri = () => {
  const [dataUri, setDataUri] = useState("");
  const [preview, setPreview] = useState("");
  const canvasRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.src = e.target.result;

        img.onload = () => {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
          canvas.width = img.width;
          canvas.height = img.height;

          ctx.drawImage(img, 0, 0);
          const uri = canvas.toDataURL("image/png");
          setDataUri(uri);
          setPreview(uri);
        };
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      <h1 className="text-2xl font-semibold">Image to Data URI Converter</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="border rounded-lg p-2"
      />
      <canvas ref={canvasRef} className="hidden"></canvas>
      {preview && (
        <>
          <img src={preview} alt="Preview" className="rounded-2xl shadow-lg max-w-md" />
          <p className="text-lg font-medium">Base64 Output:</p>
          <textarea
            readOnly
            value={dataUri}
            rows="6"
            className="w-full max-w-md p-2 border rounded-lg text-sm"
          ></textarea>
        </>
      )}
    </div>
  );
};

export default ImageToDataUri;
