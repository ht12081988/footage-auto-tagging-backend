"""
auth.py - Password hashing and operator authentication helpers
"""
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import models

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_operator_by_email(db: Session, email: str):
    return db.query(models.Operator).filter(models.Operator.email == email).first()


def create_operator(db: Session, email: str, password: str, full_name: str = "", role: str = "operator"):
    hashed = hash_password(password)
    op = models.Operator(
        email=email,
        hashed_password=hashed,
        full_name=full_name,
        role=role,
    )
    db.add(op)
    db.commit()
    db.refresh(op)
    return op


def authenticate_operator(db: Session, email: str, password: str):
    """Returns the operator if credentials match, else None."""
    op = get_operator_by_email(db, email)
    if not op:
        return None
    if not verify_password(password, op.hashed_password):
        return None
    if not op.is_active:
        return None
    return op
